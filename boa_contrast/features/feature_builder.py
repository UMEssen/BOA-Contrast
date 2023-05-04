import logging
import math
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import cc3d
import cv2
import numpy as np
import SimpleITK as sitk
from scipy import ndimage, spatial
from scipy.stats import kurtosis, skew
from skimage import draw

from boa_contrast.util.constants import INTERESTING_REGIONS, ORGANS, VERTICAL_REGIONS
from boa_contrast.util.totalseg_body_regions import REGION_MAP

logger = logging.getLogger(__name__)


class FeatureBuilder:
    def __init__(
        self,
        dataset_id: str,
        store_custom_regions: bool = False,
        one_mask_per_file: bool = False,
    ):
        self.dataset_id = dataset_id
        self.store_custom_regions = store_custom_regions
        self.one_mask_per_file = one_mask_per_file

    def write_to_nifti(
        self, mask: np.ndarray, reference: sitk.Image, output: str
    ) -> None:
        if not self.store_custom_regions:
            return
        probs_image = sitk.GetImageFromArray(mask.astype(np.uint8))
        probs_image.CopyInformation(reference)  # type: ignore
        sitk.WriteImage(probs_image, output, True)

    def compute_features(
        self,
        ct_data_path: Path,
        segmentation_path: Path,
    ) -> Optional[Dict[str, Any]]:
        if not ct_data_path.exists():
            raise ValueError(f"The CT {ct_data_path} does not exist.")
        if not segmentation_path.exists():
            return None
        samples: Dict[str, Any] = {}
        ct_image = sitk.ReadImage(str(ct_data_path))
        ct_data = sitk.GetArrayViewFromImage(ct_image)
        body_region_all, ivc_mask = None, None
        if not self.one_mask_per_file and (segmentation_path / "total.nii.gz").exists():
            body_regions = sitk.ReadImage(str(segmentation_path / "total.nii.gz"))
            body_region_all = sitk.GetArrayViewFromImage(body_regions)
            ivc_mask = np.zeros(body_region_all.shape, dtype=bool)
            ivc_mask[body_region_all == REGION_MAP["inferior_vena_cava"]] = True
        elif (segmentation_path / "inferior_vena_cava.nii.gz").exists():
            ivc_region = sitk.ReadImage(
                str(segmentation_path / "inferior_vena_cava.nii.gz")
            )
            ivc_mask = sitk.GetArrayViewFromImage(ivc_region).astype(bool)
        hepatics_veins_params = dict(
            ivc_mask=ivc_mask,  # 5 cm radius for the circle
            circle_radius_px=50 / ct_image.GetSpacing()[0],  # type: ignore
            # Compute the number of slices that are also 5 cm
            num_slices=int(math.ceil(50 / ct_image.GetSpacing()[-1])),  # type: ignore
        )
        for region in INTERESTING_REGIONS:
            if self.one_mask_per_file:
                if not (segmentation_path / f"{region}.nii.gz").exists():
                    continue
                body_regions = sitk.ReadImage(
                    str(segmentation_path / f"{region}.nii.gz")
                )
                region_mask = sitk.GetArrayViewFromImage(body_regions)
                region_mask = region_mask.astype(bool)
            else:
                assert (
                    body_region_all is not None
                ), "The segmentation total.nii.gz does not exist."
                region_mask = np.zeros(body_region_all.shape, dtype=bool)
                region_mask[body_region_all == REGION_MAP[region]] = True

            if len(np.unique(region_mask)) == 1:
                continue

            if region in ORGANS:
                region_mask = remove_small_connected_components(region_mask)

            for condition, region_name, region_fn, kwa in [
                (
                    ("kidney_" in region and np.sum(region_mask) > 2),
                    f"{region}_pelvis",
                    get_pelvis_region,
                    {},
                ),
                (
                    ("liver" in region and ivc_mask is not None),
                    "hepatic_veins",
                    get_hepatic_veins_region,
                    hepatics_veins_params,
                ),
            ]:
                if not condition:
                    continue
                try:
                    new_region = region_fn(region_mask, **kwa)  # type: ignore # TODO: Fix me
                    self.write_to_nifti(
                        new_region,
                        ct_image,
                        str(segmentation_path / f"{region_name}.nii.gz"),
                    )
                    add_stats_for_region(
                        samples=samples,
                        ct_data=ct_data,
                        region_mask=new_region,
                        region_name=region_name,
                    )
                except Exception:
                    logger.warning(
                        f"{segmentation_path.name} could not compute {region_name}"
                    )
            if region in VERTICAL_REGIONS:
                for i, partial_region_mask in enumerate(
                    create_split_regions(region_mask, n_regions=3),
                    start=1,
                ):
                    self.write_to_nifti(
                        partial_region_mask,
                        ct_image,
                        str(segmentation_path / f"{region}_part{i}.nii.gz"),
                    )
                    add_stats_for_region(
                        samples=samples,
                        ct_data=ct_data,
                        region_mask=partial_region_mask,
                        region_name=f"{region}_part{i}",
                    )
            else:
                add_stats_for_region(
                    samples=samples,
                    ct_data=ct_data,
                    region_mask=region_mask,
                    region_name=region,
                )
        liver_vessels_path = segmentation_path / "liver_vessels.nii.gz"
        if liver_vessels_path.exists():
            body_regions = sitk.ReadImage(str(liver_vessels_path))
            region_mask = sitk.GetArrayViewFromImage(body_regions)
            region_mask = region_mask.astype(bool)
            add_stats_for_region(
                samples=samples,
                ct_data=ct_data,
                region_mask=region_mask,
                region_name="liver_vessels",
            )
        assert len(samples) > 0, f"No regions were found in {segmentation_path.name}."
        return samples


def add_stats_for_region(
    samples: Dict[str, Any],
    ct_data: np.ndarray,
    region_mask: np.ndarray,
    region_name: str,
) -> None:
    if np.sum(region_mask) > 0:
        compute_statistics(
            ct_data[region_mask],
            region_name,
            samples,
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
    values: np.ndarray,
    region_name: str,
    output_dict: Dict[str, float],
) -> None:
    for func in [np.mean, np.std, np.min, np.median, np.max, skew, kurtosis]:
        output_dict[f"{region_name}_{func.__name__}"] = float(func(values))
    # Percentiles
    for p in [1, 5, 25, 75, 95, 99]:
        output_dict[f"{region_name}_{p}_percentile"] = float(np.percentile(values, p))


def remove_small_connected_components(
    mask: np.ndarray, percent_small_component: int = 1
) -> Any:
    num_positive = np.sum(mask) * (percent_small_component / 100)
    labels_out = cc3d.dust(mask, threshold=num_positive, in_place=False)
    return labels_out


def get_pelvis_region(mask: np.ndarray) -> Any:
    points = np.transpose(np.where(mask))
    hull = spatial.ConvexHull(points)
    deln = spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(mask.shape), axis=-1)  # type: ignore
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    flooded_image = np.zeros(mask.shape, dtype=bool)
    flooded_image[out_idx] = True
    final_image = np.logical_and(flooded_image, np.logical_not(mask)).astype(np.uint8)
    kernel = np.ones((4, 4), np.uint8)
    return remove_small_connected_components(
        cv2.morphologyEx(final_image, cv2.MORPH_OPEN, kernel)
    ).astype(bool)


def get_hepatic_veins_region(
    liver_mask: np.ndarray,
    ivc_mask: np.ndarray,
    circle_radius_px: int = 5,
    num_slices: int = 10,
) -> Any:
    # Find out how big the masks are at every stage
    pixel_distribution = np.sum(ivc_mask, axis=(1, 2))
    # Get the 60% percentile of that
    perc = np.percentile(pixel_distribution[pixel_distribution > 0], q=60)
    # Get the last one that is still above the percentile, so it is a proper slice
    top_ivc_slice = np.where(pixel_distribution >= perc)[0][-1]
    x_center, y_center = ndimage.center_of_mass(ivc_mask[top_ivc_slice, :, :])
    # plt.imshow(ct[top_ivc_slice, :, :], cmap="gray")
    # plt.imshow(ivc_mask[top_ivc_slice, :, :], cmap="jet", alpha=0.5)
    # plt.show()
    # exit()
    # New array
    star = np.zeros(liver_mask.shape, dtype=bool)
    # Make the circle around the found center of mass
    rr, cc = draw.disk(
        center=(x_center, y_center), radius=circle_radius_px, shape=liver_mask.shape[1:]
    )
    # Set these points to the liver values
    star[top_ivc_slice - num_slices : top_ivc_slice, rr, cc] = liver_mask[
        top_ivc_slice - num_slices : top_ivc_slice, rr, cc
    ]
    return remove_small_connected_components(star).astype(bool)
