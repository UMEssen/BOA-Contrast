import json
import logging
import unittest
from pathlib import Path
from typing import Any, cast

import pandas as pd
from totalsegmentator.map_to_binary import class_map

from boa_contrast import compute_segmentation, predict
from boa_contrast.ml import ContrastRecognition
from boa_contrast.util.totalseg_body_regions import REGION_MAP

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TestContrast(unittest.TestCase):
    def setUp(self) -> None:
        self.iv_phase = ContrastRecognition(task="iv_phase")
        self.git_c = ContrastRecognition(task="git")

    def test_prediction(self) -> None:
        with (Path.cwd() / "tests" / "example_features.json").open("r") as f:
            sample = json.load(f)
        pr_output = next(iter(self.iv_phase.predict_batch([sample])))

        self.assertEqual(pr_output["ensemble_predicted_class"], "NON_CONTRAST")

        gitr_output = next(iter(self.git_c.predict_batch([sample])))
        self.assertEqual(
            gitr_output["ensemble_predicted_class"], "NO_CONTRAST_IN_GI_TRACT"
        )

    def test_data_frame(self) -> None:
        with (Path.cwd() / "tests" / "example_features.json").open("r") as f:
            sample = json.load(f)
        df = pd.DataFrame([sample])
        pred, _ = self.iv_phase.predict(df)
        self.assertEqual(pred[0], 0)

        pred, _ = self.git_c.predict(df)
        self.assertEqual(pred[0], 0)

    def test_region_map(self) -> None:
        for k, v in REGION_MAP.items():
            self.assertEqual(class_map["total"][v], k)

    def test_compute_segmentation_without_docker(self) -> None:
        data_dir = Path.cwd() / "data"
        ct_path = data_dir / "image.nii.gz"
        for file in data_dir.glob("*.nii.gz"):
            if file.name == ct_path.name:
                continue
            file.unlink()
        compute_segmentation(ct_path, data_dir, 0)
        for file in ["liver", "liver_vessels"]:
            self.assertTrue((data_dir / f"{file}.nii.gz").is_file())

    # def test_compute_segmentation_with_docker(self) -> None:
    #     data_dir = Path.cwd() / "data"
    #     ct_path = data_dir / "image.nii.gz"
    #     for file in data_dir.glob("*.nii.gz"):
    #         if file.name == ct_path.name:
    #             continue
    #         file.unlink()
    #     compute_segmentation(ct_path, data_dir, 0, "1011", True)
    #     for file in ["liver", "liver_vessels"]:
    #         self.assertTrue((data_dir / f"{file}.nii.gz").is_file())

    def test_predict(self) -> None:
        data_dir = Path.cwd() / "data"
        ct_path = data_dir / "image.nii.gz"

        results_1 = predict(ct_path, data_dir, one_mask_per_file=True)
        results_2 = predict(ct_path, data_dir, one_mask_per_file=False)

        self.assertIsNotNone(results_1)
        self.assertIsNotNone(results_2)

        results_1 = cast(dict[str, Any], results_1)
        results_2 = cast(dict[str, Any], results_2)

        self.assertDictEqual(results_1, results_2)


if __name__ == "__main__":
    unittest.main()
