import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from boa_contrast.features import FeatureBuilder
from boa_contrast.ml import ContrastRecognition
from boa_contrast.util.constants import Contrast_in_GI, IVContrast

logger = logging.getLogger(__name__)


def predict(
    ct_path: Union[Path, str],
    segmentation_folder: Union[Path, str],
    phase_model_name: str = "real_IV_class_HistGradientBoostingClassifier_5class_2023-04-07",
    git_model_name: str = "KM_in_GI_HistGradientBoostingClassifier_2class_2023-04-07",
) -> Optional[Dict[str, Any]]:
    # Download data for model
    curr_dir, _ = os.path.split(__file__)
    model_folder = Path(str(curr_dir)) / "models"
    ct_path = Path(ct_path)
    logger.info("Computing the features...")
    start = time.time()
    fb = FeatureBuilder(dataset_id="inference", one_mask_per_file=True)
    sample = fb.compute_features(
        ct_data_path=ct_path,
        segmentation_path=Path(segmentation_folder),
    )
    logger.info(f"Features computed in {time.time() - start:0.5f}s")
    if sample is None:
        logger.warning("The segmentation does not exist.")
        return None

    logger.info("Computing the contrast phase prediction...")

    pr_phase = ContrastRecognition(
        output_classes=IVContrast,
        label_column="real_IV_class",
        feature_columns=None,
    )
    logger.info(f"Using model {model_folder / phase_model_name}")
    pr_phase.load_models(model_folder / phase_model_name)
    logger.info("Model loaded")
    start = time.time()
    pr_output = list(pr_phase.predict_batch([sample]))[0]
    logger.info(f"Phase prediction computed in {time.time() - start:0.5f}s")

    logger.info("Computing the GIT contrast prediction...")
    gitr = ContrastRecognition(
        output_classes=Contrast_in_GI,
        label_column="KM_in_GI",
        feature_columns=None,
    )
    logger.info(f"Using model {model_folder / git_model_name}")
    gitr.load_models(model_folder / git_model_name)
    logger.info("Model loaded")
    start = time.time()
    gitr_output = list(gitr.predict_batch([sample]))[0]
    logger.info(f"GIT prediction computed in {time.time() - start:0.5f}s")

    return dict(
        **{"phase_" + key: value for key, value in pr_output.items()},
        **{"git_" + key: value for key, value in gitr_output.items()},
    )


def compute_segmentation(
    ct_path: Path,
    segmentation_folder: Union[Path, str],
    device_id: Optional[int],
    user_id: Optional[str],
    compute_with_docker: bool,
) -> Path:
    segmentation_folder = Path(segmentation_folder)
    example_output = segmentation_folder / "liver.nii.gz"
    vessels_output = segmentation_folder / "liver_vessels.nii.gz"
    tasks = []
    if example_output.exists():
        logger.info("The full body segmentation exists and will not be recomputed.")
    else:
        tasks = ["total"]

    if vessels_output.exists():
        logger.info("The liver vessels segmentation exists and will not be recomputed.")
    else:
        tasks.append("liver_vessels")

    if example_output.exists() and vessels_output.exists():
        return segmentation_folder

    logger.info("Segmentation is being computed")
    if compute_with_docker:
        # TODO: Set the docker image to something more stable
        command = (
            "docker run "
            + (f"--user {user_id}:{user_id} " if user_id is not None else "")
            + "--rm "
            + (f"--gpus device={device_id} " if device_id is not None else "")
            + "--ipc=host "
            f"-v {ct_path}:/image.nii.gz "
            f"-v {segmentation_folder}:/output "
            "wasserth/totalsegmentator_container:master "
            "TotalSegmentator -i /image.nii.gz -o /output"
        )
        subprocess.run(
            command.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            # capture_output=True,
            check=True,
            universal_newlines=True,
        )
        logger.info("Segmentation done")
    else:
        from totalsegmentator.python_api import totalsegmentator

        for task in tasks:
            logger.info(f"Computing segmentation for task {task}")
            totalsegmentator(
                input=ct_path,
                output=segmentation_folder,
                task=task,
                ml=False,
                preview=False,
                force_split=False,
                nora_tag="None",
                quiet=False,
                verbose=0,
                test=0,
                crop_path=None,
            )
        logger.info("Segmentations done")

    return segmentation_folder
