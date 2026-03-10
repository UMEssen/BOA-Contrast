import json
import logging
import unittest
from pathlib import Path

import pandas as pd

from boa_contrast.ml import ContrastRecognition

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TestContrast(unittest.TestCase):
    def setUp(self) -> None:
        self.iv_phase = ContrastRecognition(task="iv_phase")
        self.git_c = ContrastRecognition(task="git")

    @staticmethod
    def test_imports() -> None:
        from boa_contrast import compute_segmentation, predict  # noqa

    def test_prediction(self) -> None:
        with (Path("tests") / "example_features.json").open("r") as f:
            sample = json.load(f)
        pr_output = next(iter(self.iv_phase.predict_batch([sample])))

        self.assertEqual(pr_output["ensemble_predicted_class"], "NON_CONTRAST")

        gitr_output = next(iter(self.git_c.predict_batch([sample])))
        self.assertEqual(
            gitr_output["ensemble_predicted_class"], "NO_CONTRAST_IN_GI_TRACT"
        )

    def test_data_frame(self) -> None:
        with (Path("tests") / "example_features.json").open("r") as f:
            sample = json.load(f)
        df = pd.DataFrame([sample])
        pred, _ = self.iv_phase.predict(df)
        self.assertEqual(pred[0], 0)

        pred, _ = self.git_c.predict(df)
        self.assertEqual(pred[0], 0)


if __name__ == "__main__":
    unittest.main()
