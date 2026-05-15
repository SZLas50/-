from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[RandomForestClassifier, LogisticRegression]:
    """
    Train Random Forest and Logistic Regression models on the training data.

    This function initializes and fits two classification models: a Random Forest
    and a Logistic Regression model, using the provided training features and labels.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector (binary labels).

    Returns:
        tuple[RandomForestClassifier, LogisticRegression]: Trained Random Forest and Logistic Regression models.
    """
    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return rf, lr


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """
    Evaluate a trained model on test data and compute key metrics.

    This function predicts labels and probabilities on the test set, then calculates
    accuracy, AUC, and a detailed classification report.

    Args:
        model: Trained sklearn model with predict and predict_proba methods.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test target vector.

    Returns:
        dict[str, float]: Dictionary containing 'accuracy', 'auc', and 'report' (classification report as dict).
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "auc": auc,
        "report": report,
    } # type: ignore


def save_model(model, path: str) -> None:
    """
    Save a trained model to a file using joblib.

    Args:
        model: The trained sklearn model to save.
        path (str): File path where to save the model.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    """
    Load a trained model from a file using joblib.

    Args:
        path (str): File path from where to load the model.

    Returns:
        The loaded sklearn model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    return joblib.load(path)


def compare_models(
    rf: RandomForestClassifier,
    lr: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    """
    Compare the performance of Random Forest and Logistic Regression models.

    This function evaluates both models on the test set and returns their metrics.

    Args:
        rf (RandomForestClassifier): Trained Random Forest model.
        lr (LogisticRegression): Trained Logistic Regression model.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test target vector.

    Returns:
        dict[str, dict[str, float]]: Dictionary with keys 'random_forest' and 'logistic_regression',
                                      each containing evaluation metrics.
    """
    return {
        "random_forest": evaluate_model(rf, X_test, y_test),
        "logistic_regression": evaluate_model(lr, X_test, y_test),
    }
