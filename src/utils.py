import os
import pickle
import sys

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exceptions import CustomException


def save_object(file_path: str, obj) -> None:
    """Serialise un objet Python sur disque."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """Recharge un objet serialise par save_object."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict) -> dict:
    """
    Entraine chaque modele avec recherche d'hyperparametres.

    Renvoie un dict : nom -> {"train_r2", "test_r2", "best_params", "model"}.
    Le modele renvoye est celui reellement entraine, ce qui evite d'aller
    le rechercher dans le dictionnaire d'origine.
    """
    try:
        report = {}

        for nom, model in models.items():
            grille = params.get(nom, {})

            search = GridSearchCV(model, grille, cv=3)
            search.fit(X_train, y_train)

            model.set_params(**search.best_params_)
            model.fit(X_train, y_train)

            report[nom] = {
                "train_r2": r2_score(y_train, model.predict(X_train)),
                "test_r2": r2_score(y_test, model.predict(X_test)),
                "best_params": search.best_params_,
                "model": model,
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)