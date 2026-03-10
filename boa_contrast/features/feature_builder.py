import logging
import traceback
from collections.abc import Generator
from pathlib import Path
from typing import Any

import cc3d
import cv2
import numpy as np
import SimpleITK as sitk
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import kurtosis, skew

from boa_contrast.util.constants import INTERESTING_REGIONS, ORGANS, VERTICAL_REGIONS
from boa_contrast.util.totalseg_body_regions import REGION_MAP

logger = logging.getLogger(__name__)


class FeatureBuilder:
    PELVIS_KERNEL = np.ones((4, 4), np.uint8)

    def __init__(
        self,
        dataset_id: str,
        store_custom_regions: bool = False,
        one_mask_per_file: bool = False,
        total_segmentation_name: str = "total.nii.gz",
        label_map: dict[str, Any] | None = REGION_MAP,
    ):
        self.dataset_id = dataset_id
        self.store_custom_regions = store_custom_regions
        self.one_mask_per_file = one_mask_per_file
        self.total_segmentation_name = total_segmentation_name
        self.label_map = label_map if label_map is not None else REGION_MAP

    def write_to_nifti(
        self, mask: np.ndarray, reference: sitk.Image, output: str
    ) -> None:
        if not self.store_custom_regions:
            return
        probs_image = sitk.GetImageFromArray(mask.astype(np.uint8))
        probs_image.CopyInformation(reference)
        sitk.WriteImage(probs_image, output, True)

    def compute_features(  # noqa: C901
        self,
        ct_data_path: Path,
        segmentation_path: Path,
    ) -> dict[str, Any] | None:
        if not ct_data_path.is_file():
            raise FileNotFoundError(f"The CT {ct_data_path} does not exist.")
        if not segmentation_path.is_dir():
            return None
        samples: dict[str, Any] = {}
        ct_image = sitk.ReadImage(ct_data_path)
        ct_data = sitk.GetArrayViewFromImage(ct_image)
        if ct_data.ndim != 3:
            raise ValueError("The data should be a 3D CT scan.")

        body_region_all = None
        if (
            not self.one_mask_per_file
            and (segmentation_path / self.total_segmentation_name).is_file()
        ):
            body_regions = sitk.ReadImage(
                segmentation_path / self.total_segmentation_name
            )
            body_region_all = sitk.GetArrayFromImage(body_regions)
            del body_regions

        for region in INTERESTING_REGIONS:
            if self.one_mask_per_file:
                if not (segmentation_path / f"{region}.nii.gz").is_file():
                    continue
                body_regions = sitk.ReadImage(segmentation_path / f"{region}.nii.gz")
                region_mask = sitk.GetArrayViewFromImage(body_regions).astype(bool)
                del body_regions
            else:
                if body_region_all is None:
                    raise FileNotFoundError(
                        f"The segmentation {self.total_segmentation_name}"
                        " does not exist."
                    )
                if isinstance(self.label_map[region], int):
                    region_mask = body_region_all == self.label_map[region]
                else:
                    region_mask = np.isin(body_region_all, self.label_map[region])
            if np.sum(region_mask) == 0:
                continue
            region_mask, ct_data_region = crop_mask(region_mask, ct_data)
            if region in ORGANS:
                remove_small_connected_components(region_mask)

            if "kidney_" in region and np.sum(region_mask) > 2:
                try:
                    new_region = get_pelvis_region(region_mask)
                    compute_statistics(
                        samples,
                        ct_data_region[new_region],
                        region_name=f"{region}_pelvis",
                    )
                except Exception:
                    traceback.print_exc()
                    logger.warning(
                        f"{segmentation_path.name} could not compute {region}_pelvis"
                    )
            if region in VERTICAL_REGIONS:
                for i, partial_region_mask in enumerate(
                    create_split_regions(region_mask, n_regions=3),
                    start=1,
                ):
                    compute_statistics(
                        samples,
                        ct_data_region[partial_region_mask],
                        region_name=f"{region}_part{i}",
                    )
            else:
                compute_statistics(
                    samples,
                    ct_data_region[region_mask],
                    region_name=region,
                )

        liver_vessels_path = segmentation_path / "liver_vessels.nii.gz"
        if liver_vessels_path.is_file():
            body_regions = sitk.ReadImage(liver_vessels_path)
            region_mask = sitk.GetArrayViewFromImage(body_regions).astype(bool)
            region_mask, ct_data_region = crop_mask(region_mask, ct_data)
            compute_statistics(
                samples,
                ct_data_region[region_mask],
                region_name="liver_vessels",
            )
        if not samples:
            raise RuntimeError(f"No regions were found in {segmentation_path.name}.")
        return samples


def crop_mask(mask: np.ndarray, ct_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    x_min, y_min, z_min = [coords[:, i].min() for i in range(3)]
    x_max, y_max, z_max = [coords[:, i].max() + 1 for i in range(3)]

    return (
        mask[x_min:x_max, y_min:y_max, z_min:z_max],
        ct_data[x_min:x_max, y_min:y_max, z_min:z_max],
    )


def create_split_regions(
    mask: np.ndarray, n_regions: int = False
) -> Generator[np.ndarray, None, None]:
    slices = np.where(mask.any(axis=(1, 2)))[0]
    points = [
        int(sl)
        for sl in np.linspace(
            start=slices.min(), stop=slices.max() + 1, num=n_regions + 1
        )
    ]
    for i in range(len(points) - 1):
        new_mask = np.zeros(mask.shape, dtype=bool)
        start, end = points[i : i + 2]
        new_mask[start:end, :, :] = mask[start:end, :, :]
        yield new_mask


def compute_statistics(
    output_dict: dict[str, Any],
    values: np.ndarray,
    region_name: str,
) -> None:
    if len(values) == 0:
        return
    for func in [np.mean, np.std, np.min, np.median, np.max, skew, kurtosis]:
        output_dict[f"{region_name}_{func.__name__}"] = float(func(values))
    # Percentiles
    for p in [5, 25, 75, 95]:
        output_dict[f"{region_name}_{p}_percentile"] = float(np.percentile(values, p))


def remove_small_connected_components(
    mask: np.ndarray, percent_small_component: int = 1
) -> None:
    num_positive = np.sum(mask) * (percent_small_component / 100)
    cc3d.dust(mask, threshold=num_positive, in_place=True)


def get_pelvis_region(mask: np.ndarray) -> Any:
    # Get points and build convex hull around the kidneys
    points = np.transpose(np.where(mask))
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices])
    # Get valid indices, this is faster than np.indices + np.stack
    idx = np.mgrid[: mask.shape[0], : mask.shape[1], : mask.shape[2]].transpose(
        1, 2, 3, 0
    )
    # Flood withing the convex hull of the image
    flooded_image = np.zeros(mask.shape, dtype=bool)
    # TODO: find_simplex is currently the slowest part, can we speed it up?
    flooded_image[deln.find_simplex(idx) > -1] = True
    # Get final_image using boolean operations
    final_image = np.logical_and(flooded_image, np.logical_not(mask)).astype(np.uint8)
    final_image = cv2.morphologyEx(
        final_image, cv2.MORPH_OPEN, FeatureBuilder.PELVIS_KERNEL
    )
    # Remove small components
    remove_small_connected_components(final_image)
    return final_image.astype(bool)
