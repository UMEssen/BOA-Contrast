import enum
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ContrastRecognition:
    CONFIDENCE_THRESHOLD = 0.85

    def __init__(
        self,
        output_classes: Type[enum.IntEnum],
        feature_columns: Optional[List[str]],
        label_column: str,
    ):
        self.output_classes = output_classes
        self.n_classes = len(output_classes)
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.models: List[Pipeline] = []

    def load_models(self, model_folder: Path) -> None:
        if not model_folder.exists():
            raise ValueError(f"The given model folder {model_folder} does not exist.")
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
            data = data[self.feature_columns].to_numpy()
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
