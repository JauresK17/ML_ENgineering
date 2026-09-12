import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

SCORE_MINIMUM = 0.6


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def get_models(self) -> dict:
        """Renvoie les modeles candidats, indexes par nom."""
        return {
            "Random Forest": RandomForestRegressor(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Linear Regression": LinearRegression(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "XGBRegressor": XGBRegressor(random_state=42),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_state=42),
            "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
        }

    def get_params(self) -> dict:
        """Grilles d'hyperparametres. Les cles doivent correspondre a get_models()."""
        return {
            "Random Forest": {
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Decision Tree": {
                "criterion": [
                    "squared_error",
                    "friedman_mse",
                    "absolute_error",
                    "poisson",
                ],
            },
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.05, 0.01, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Linear Regression": {},
            "K-Neighbors Regressor": {
                "n_neighbors": [5, 7, 9, 11],
            },
            "XGBRegressor": {
                "learning_rate": [0.1, 0.05, 0.01, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "CatBoosting Regressor": {
                "depth": [6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "iterations": [30, 50, 100],
            },
            "AdaBoost Regressor": {
                "learning_rate": [0.5, 0.1, 0.01, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
        }

    def initiate_model_training(self, train_array, test_array) -> float:
        try:
            logging.info("Separation des features et de la cible")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = self.get_models()
            params = self.get_params()

            # Garde-fou : une cle manquante d'un cote fait planter le GridSearch.
            ecart = set(models) ^ set(params)
            if ecart:
                raise CustomException(
                    f"Cles non alignees entre models et params : {sorted(ecart)}", sys
                )

            report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            best_name = max(report, key=lambda nom: report[nom]["test_r2"])
            best = report[best_name]

            logging.info(
                f"Meilleur modele : {best_name} "
                f"(train_r2={best['train_r2']:.4f}, test_r2={best['test_r2']:.4f})"
            )

            if best["test_r2"] < SCORE_MINIMUM:
                raise CustomException(
                    f"Aucun modele satisfaisant : meilleur R2 = {best['test_r2']:.4f}",
                    sys,
                )

            best_model = best["model"]

            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model,
            )

            y_pred = best_model.predict(X_test)
            return r2_score(y_test, y_pred)

        except Exception as e:
            raise CustomException(e, sys)