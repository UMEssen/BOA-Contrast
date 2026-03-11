import json
import logging
import os
import unittest
from pathlib import Path
from typing import Any, cast

import pandas as pd
from totalsegmentator.map_to_binary import class_map

from boa_contrast import compute_segmentation, default, predict
from boa_contrast.ml import ContrastRecognition
from boa_contrast.utils.totalseg_body_regions import REGION_MAP

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

    @unittest.skipIf(os.getenv("CI") == "true", "Skipped in CI")
    def test_workflow(self) -> None:
        data_dir = Path.cwd() / "data"
        mapping_df = pd.read_csv(data_dir / "mapping.csv")
        for row in mapping_df.itertuples(index=False):
            folder = data_dir / row.folder_name
            json_file = folder / "prediction.json"
            ct_path = next(folder.glob("*.nii.gz"))
            seg_dir = folder / "segmentations"
            compute_segmentation(ct_path, seg_dir, 0)

            for name in ("liver", "liver_vessels"):
                self.assertTrue((seg_dir / f"{name}.nii.gz").is_file())

            result_dict = predict(ct_path, seg_dir)

            self.assertIsNotNone(result_dict)

            result_dict = cast(dict[str, Any], result_dict)
            json_file.write_text(
                json.dumps(result_dict, indent=2, default=default), "utf-8"
            )
            phase_class = result_dict["phase_ensemble_predicted_class"]
            git_class = result_dict["git_ensemble_predicted_class"]

            self.assertEqual(phase_class, row.phase_class)
            self.assertEqual(git_class, row.git_class)


if __name__ == "__main__":
    unittest.main()
