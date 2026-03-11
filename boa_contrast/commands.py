import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from boa_contrast.features import FeatureBuilder
from boa_contrast.ml import ContrastRecognition

logger = logging.getLogger(__name__)


def predict(
    ct_path: Path | str,
    segmentation_folder: Path | str,
    phase_model_name: str = "real_IV_class_HistGradientBoostingClassifier_5class_2023-07-20",
    git_model_name: str = "KM_in_GI_HistGradientBoostingClassifier_2class_2023-07-18",
    one_mask_per_file: bool = True,
    store_custom_regions: bool = False,
    total_segmentation_name: str = "total.nii.gz",
    label_map: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    # Download data for model
    ct_path = Path(ct_path)
    logger.info("Computing the features...")
    start = time.time()
    fb = FeatureBuilder(
        dataset_id="inference",
        one_mask_per_file=one_mask_per_file,
        store_custom_regions=store_custom_regions,
        total_segmentation_name=total_segmentation_name,
        label_map=label_map,
    )
    sample = fb.compute_features(
        ct_data_path=ct_path,
        segmentation_path=Path(segmentation_folder),
    )
    logger.info(f"Features computed in {time.time() - start:0.5f}s")
    if sample is None:
        logger.warning("The segmentation does not exist.")
        return None

    logger.info("Computing the contrast phase prediction...")

    pr_phase = ContrastRecognition(task="iv_phase", model_name=phase_model_name)
    start = time.time()
    pr_output = next(iter(pr_phase.predict_batch([sample])))
    logger.info(f"Phase prediction computed in {time.time() - start:0.5f}s")

    logger.info("Computing the GIT contrast prediction...")
    gitr = ContrastRecognition(task="git", model_name=git_model_name)

    start = time.time()
    gitr_output = next(iter(gitr.predict_batch([sample])))
    logger.info(f"GIT prediction computed in {time.time() - start:0.5f}s")

    return {
        **{"phase_" + key: value for key, value in pr_output.items()},
        **{"git_" + key: value for key, value in gitr_output.items()},
    }


def compute_segmentation(
    ct_path: Path,
    segmentation_folder: Path | str,
    device_id: int | None = None,
    user_id: str | None = None,
    compute_with_docker: bool = False,
) -> Path:
    segmentation_folder = Path(segmentation_folder)
    example_output = segmentation_folder / "liver.nii.gz"
    vessels_output = segmentation_folder / "liver_vessels.nii.gz"
    tasks = []
    if example_output.is_file():
        logger.info("The full body segmentation exists and will not be recomputed.")
    else:
        tasks = ["total"]

    if vessels_output.is_file():
        logger.info("The liver vessels segmentation exists and will not be recomputed.")
    else:
        tasks.append("liver_vessels")

    if not tasks:
        return segmentation_folder

    logger.info("Segmentation is being computed")
    # TODO: Make the crop region liver findable by the totalsegmentator if multilabel is true
    if compute_with_docker:
        logger.info("Using docker.")
        for task in tasks:
            logger.info(f"Computing segmentation for task {task}")
            command = (
                "docker run "
                + (f"--user {user_id}:{user_id} " if user_id is not None else "")
                + "--rm "
                + (f"--gpus device={device_id} " if device_id is not None else "")
                + "--ipc=host "
                f"-v {ct_path.absolute()}:/image.nii.gz "
                f"-v {segmentation_folder.absolute()}:/output "
                "wasserth/totalsegmentator:2.12.0 "
                f"TotalSegmentator -i /image.nii.gz -o /output -ta {task}"
            )
            start = time.time()
            subprocess.run(  # noqa: S603
                command.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                check=True,
                text=True,
            )
            logger.info(
                f"Segmentation computed for {task} in {time.time() - start:0.5f}s"
            )
    else:
        logger.info("Using the TotalSegmentator package.")

        from totalsegmentator.python_api import totalsegmentator  # noqa

        for task in tasks:
            logger.info(f"Computing segmentation for task {task}")
            start = time.time()
            totalsegmentator(
                input=ct_path,
                output=segmentation_folder,
                task=task,
            )
            logger.info(
                f"Segmentation computed for {task} in {time.time() - start:0.5f}s"
            )

    return segmentation_folder
