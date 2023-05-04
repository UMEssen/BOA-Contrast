import json
import unittest
from pathlib import Path

from boa_contrast.ml import ContrastRecognition
from boa_contrast.util.constants import Contrast_in_GI, IVContrast


class BasicTests(unittest.TestCase):
    @staticmethod
    def testImports() -> None:
        from boa_contrast import compute_segmentation, predict  # noqa: F401

    def testPrediction(self) -> None:
        with (Path("tests") / "example_features.json").open("r") as f:
            sample = json.load(f)

        model_folder = Path("boa_contrast") / "models"
        pr_phase = ContrastRecognition(
            output_classes=IVContrast,
            label_column="real_IV_class",
            feature_columns=None,
        )
        pr_phase.load_models(
            model_folder
            / "real_IV_class_HistGradientBoostingClassifier_5class_2023-04-07"
        )
        pr_output = list(pr_phase.predict_batch([sample]))[0]

        assert pr_output["ensemble_predicted_class"] == "NON_CONTRAST"
        gitr = ContrastRecognition(
            output_classes=Contrast_in_GI,
            label_column="KM_in_GI",
            feature_columns=None,
        )
        gitr.load_models(
            model_folder / "KM_in_GI_HistGradientBoostingClassifier_2class_2023-04-07"
        )
        gitr_output = list(gitr.predict_batch([sample]))[0]
        assert gitr_output["ensemble_predicted_class"] == "NO_CONTRAST_IN_GI_TRACT"


if __name__ == "__main__":
    unittest.main()
