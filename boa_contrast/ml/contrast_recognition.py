import enum
import logging
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from boa_contrast.util.constants import ContrastInGI, IVContrast

logger = logging.getLogger(__name__)


class ContrastRecognition:
    CONFIDENCE_THRESHOLD = 0.85

    def __init__(
        self,
        task: str,
        model_name: str | None = None,
    ):
        self.output_classes: type[enum.IntEnum]
        if task.lower() == "iv_phase":
            self.output_classes = IVContrast
            if model_name is None:
                model_name = (
                    "real_IV_class_HistGradientBoostingClassifier_5class_2023-07-20"
                )
        elif task == "git":
            self.output_classes = ContrastInGI
            if model_name is None:
                model_name = "KM_in_GI_HistGradientBoostingClassifier_2class_2023-07-18"
        else:
            raise ValueError(
                f"The task {task} does not exist, it should be either IV_phase or GIT."
            )
        self.n_classes = len(self.output_classes)
        self.feature_columns: list[str] = []
        self.models: list[Pipeline] = []
        curr_dir, _ = os.path.split(__file__)
        model_folder = Path(str(curr_dir)).parent / "models"
        self.load_models(model_folder / model_name)

    def load_models(self, model_folder: Path) -> None:
        if not model_folder.exists():
            raise ValueError(f"The given model folder {model_folder} does not exist.")
        logger.info(f"Using model {model_folder}...")
        self.feature_columns = joblib.load(next(iter(model_folder.glob("features_*"))))
        self.models = [
            joblib.load(model_path)
            for model_path in sorted(
                model_folder.glob("model_*"),
                key=lambda x: x.name.split("_")[-1],
            )
        ]

    def predict(self, data: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(data, pd.DataFrame):
            if not isinstance(self.feature_columns, list):
                raise ValueError(
                    "`feature_columns` must be of type `list`, "
                    f"got `{type(self.feature_columns).__name__}`."
                )
            missing_cols = list(set(self.feature_columns) - set(data.columns))
            filled_data = data.copy()
            filled_data.loc[:, missing_cols] = np.nan
            data = filled_data[self.feature_columns].to_numpy()
        cv_proba = np.zeros((data.shape[0], self.n_classes, len(self.models)))
        for cv_id, model in enumerate(self.models):
            with threadpool_limits(limits=1, user_api="openmp"):
                test_proba = model.predict_proba(data)
            for label in range(0, self.n_classes):
                # The order in the model may be different
                label_pos = np.where(model.classes_ == label)[0][0]
                cv_proba[:, label, cv_id - 1] = test_proba[:, label_pos]

        ensemble_probas = cv_proba.mean(axis=2)
        ensemble_prediction = np.argmax(ensemble_probas, axis=1)
        return ensemble_prediction, ensemble_probas

    def predict_batch(
        self, samples: list[dict[str, Any]]
    ) -> Generator[dict[str, Any], None, None]:
        if self.feature_columns is None:
            raise ValueError("The feature columns should be existing before prediction")
        data = np.array(
            [
                [
                    features.get(feature_name, np.nan)
                    for feature_name in self.feature_columns
                ]
                for features in samples
            ]
        )
        ensemble_predictions, ensemble_probass = self.predict(data)
        for ensemble_prediction, ensemble_probas in zip(
            ensemble_predictions, ensemble_probass, strict=True
        ):
            yield {
                "ensemble_prediction": ensemble_prediction,
                "ensemble_predicted_class": self.output_classes(
                    ensemble_prediction
                ).name,
                "ensemble_probas": ensemble_probas,
            }
