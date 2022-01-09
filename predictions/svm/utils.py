from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    auc,
    roc_curve,
)
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


def perform_statistics_regression(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    interval: int,
    period: int,
) -> None:
    log = open(f"predictions/logs/regression/{model_type}_log.txt", "a")
    stats_dict = {}
    stats_dict["mrse"] = np.sqrt(mean_squared_error(y_pred, y_test))
    stats_dict["mae"] = mean_absolute_error(y_test, y_pred)
    stats_dict["mape"] = mean_absolute_percentage_error(y_test, y_pred)

    print(f"{model_type} regression, interval {interval} period {period}", file=log)
    for stat_name, stat_val in stats_dict.items():
        print(f"{stat_name} is {stat_val}", file=log)


def perform_statistics_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_type: str,
    interval: int,
    period: int,
) -> None:
    log = open(f"predictions/logs/classification/{model_type}_log.txt", "a")
    stats_dict = {}
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    stats_dict["specificity"] = tn / (tn + fp)
    stats_dict["f1"] = f1_score(y_true, y_pred)
    stats_dict["accuracy"] = accuracy_score(y_true, y_pred)
    stats_dict["precision"] = precision_score(y_true, y_pred)
    stats_dict["recall"] = recall_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    stats_dict["area under curve"] = auc(fpr, tpr)

    print(f"{model_type} classification, interval {interval} period {period}", file=log)
    for stat_name, stat_val in stats_dict.items():
        print(f"{stat_name} is {stat_val}", file=log)
