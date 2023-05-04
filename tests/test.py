import json
import unittest
from pathlib import Path

import pandas as pd

from boa_contrast.ml import ContrastRecognition
from boa_contrast.util.constants import Contrast_in_GI, IVContrast


class BasicTests(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        model_folder = Path("boa_contrast") / "models"

        self.iv_phase = ContrastRecognition(
            output_classes=IVContrast,
            label_column="real_IV_class",
            feature_columns=None,
        )
        self.iv_phase.load_models(
            model_folder
            / "real_IV_class_HistGradientBoostingClassifier_5class_2023-04-07"
        )

        self.git_c = ContrastRecognition(
            output_classes=Contrast_in_GI,
            label_column="KM_in_GI",
            feature_columns=None,
        )
        self.git_c.load_models(
            model_folder / "KM_in_GI_HistGradientBoostingClassifier_2class_2023-04-07"
        )

    @staticmethod
    def testImports() -> None:
        from boa_contrast import compute_segmentation, predict  # noqa: F401

    def testPrediction(self) -> None:
        with (Path("tests") / "example_features.json").open("r") as f:
            sample = json.load(f)
        pr_output = list(self.iv_phase.predict_batch([sample]))[0]

        assert pr_output["ensemble_predicted_class"] == "NON_CONTRAST", pr_output

        gitr_output = list(self.git_c.predict_batch([sample]))[0]
        assert (
            gitr_output["ensemble_predicted_class"] == "NO_CONTRAST_IN_GI_TRACT"
        ), gitr_output

    def testDataFrame(self) -> None:
        with (Path("tests") / "example_features.json").open("r") as f:
            sample = json.load(f)
        df = pd.DataFrame([sample])
        pred, _ = self.iv_phase.predict(df)

        assert pred[0] == 0, pred

        pred, _ = self.git_c.predict(df)
        assert pred[0] == 0, pred


if __name__ == "__main__":
    unittest.main()
