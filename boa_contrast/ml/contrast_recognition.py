import enum
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from boa_contrast.util.constants import Contrast_in_GI, IVContrast

logger = logging.getLogger(__name__)


class ContrastRecognition:
    CONFIDENCE_THRESHOLD = 0.85

    def __init__(
        self,
        task: str,
        model_name: Optional[str] = None,
    ):
        self.output_classes: Type[enum.IntEnum]
        if task.lower() == "iv_phase":
            self.output_classes = IVContrast
            if model_name is None:
                model_name = (
                    "real_IV_class_HistGradientBoostingClassifier_5class_2023-04-07"
                )
        elif task == "git":
            self.output_classes = Contrast_in_GI
            if model_name is None:
                model_name = "KM_in_GI_HistGradientBoostingClassifier_2class_2023-04-07"
        else:
            raise ValueError(
                f"The task {task} does not exist, it should be either IV_phase or GIT."
            )
        self.n_classes = len(self.output_classes)
        self.feature_columns: List[str] = []
        self.models: List[Pipeline] = []
        curr_dir, _ = os.path.split(__file__)
        model_folder = Path(str(curr_dir)).parent / "models"
        self.load_models(model_folder / model_name)

    def load_models(self, model_folder: Path) -> None:
        if not model_folder.exists():
            raise ValueError(f"The given model folder {model_folder} does not exist.")
        logger.info(f"Using model {model_folder}...")
        self.feature_columns = joblib.load(list(model_folder.glob("features_*"))[0])
        self.models = [
            joblib.load(model_path)
            for model_path in sorted(
                model_folder.glob("model_*"), key=lambda x: x.name.split("_")[-1]  # type: ignore
            )
        ]

    def predict(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(data, pd.DataFrame):
            assert isinstance(self.feature_columns, List)
            missing_cols = list(set(self.feature_columns) - set(data.columns))
            filled_data = data.copy()
            filled_data.loc[:, missing_cols] = np.nan
            data = filled_data[self.feature_columns].to_numpy()
        cv_proba = np.zeros((data.shape[0], self.n_classes, len(self.models)))
        for cv_id, model in enumerate(self.models):
            test_proba = model.predict_proba(data)
            for label in range(0, self.n_classes):
                # The order in the model may be different
                label_pos = np.where(model.classes_ == label)[0][0]
                cv_proba[:, label, cv_id - 1] = test_proba[:, label_pos]

        ensemble_probas = cv_proba.mean(axis=2)
        ensemble_prediction = np.argmax(ensemble_probas, axis=1)
        return ensemble_prediction, ensemble_probas

    def predict_batch(
        self, samples: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        assert (
            self.feature_columns is not None
        ), "The feature columns should be existing before prediction"
        data = np.array(
            [
                [
                    features[feature_name] if feature_name in features else np.nan
                    for feature_name in self.feature_columns
                ]
                for features in samples
            ]
        )
        ensemble_predictions, ensemble_probass = self.predict(data)
        for ensemble_prediction, ensemble_probas in zip(
            ensemble_predictions, ensemble_probass
        ):
            yield {
                "ensemble_prediction": ensemble_prediction,
                "ensemble_predicted_class": self.output_classes(
                    ensemble_prediction
                ).name,
                "ensemble_probas": ensemble_probas,
            }
