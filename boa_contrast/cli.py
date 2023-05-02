import argparse
import logging
from pathlib import Path

from boa_contrast.commands import compute_segmentation, predict


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ct-path",
        type=Path,
        required=True,
        help="Path to the CT scan from which the contrast information should be extracted.",
    )
    parser.add_argument(
        "--segmentation-folder",
        type=Path,
        required=True,
        help="Path to the folder where the segmentation masks should be saved.",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Whether the TotalSegmentator should be run in a Docker container.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        required=False,
        help="The ID of the user to run the docker container, otherwise the container will be run as root.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        required=False,
        help="The ID of the GPU to use, otherwise the CPU will be used.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print some additional output information.",
    )
    return parser


def run() -> None:
    parser = get_parser()
    args = parser.parse_args()
    logger = logging.getLogger()
    if args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    segmentation_folder = compute_segmentation(
        ct_path=args.ct_path,
        segmentation_folder=args.segmentation_folder,
        device_id=args.device_id,
        user_id=args.user_id,
        compute_with_docker=args.docker,
    )
    prediction = predict(
        ct_path=args.ct_path,
        segmentation_folder=segmentation_folder,
    )
    if prediction is not None:
        print(f"IV Phase: {prediction['phase_ensemble_predicted_class']}")
        print(f"Contrast in GIT: {prediction['git_ensemble_predicted_class']}")
    else:
        print("No segmentation was found.")
