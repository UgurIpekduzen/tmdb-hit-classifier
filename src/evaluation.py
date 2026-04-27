from __future__ import annotations

from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, fbeta_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve,
    classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)

_CV_KEY_MAP: dict[str, dict[str, str]] = {
    "binary":     {"F2": "f2", "PR-AUC": "avg_prec", "ROC-AUC": "roc_auc", "F1": "f1",
                   "Precision": "precision", "Recall": "recall"},
    "multiclass": {"ROC-AUC": "roc_auc", "F1": "f1",
                   "Precision": "precision", "Recall": "recall"},
    "regression": {"RMSE": "rmse", "MAE": "mae", "R2": "r2"},
    "timeseries": {"RMSE": "rmse", "MAE": "mae", "R2": "r2"},
    "multilabel": {"F1": "f1", "Precision": "precision", "Recall": "recall"},
}


def evaluate_test(
    model,
    X_test,
    y_test,
    task_type: str,
    model_name: str = "",
    cv_metrics: dict | None = None,
    experiment_name: str = "",
    feature_version: str = "",
    features: list | None = None,
    extra_tags: dict | None = None,
) -> dict:
    """
    Task type'a göre test metriklerini hesaplar, grafikleri çizer ve metrikleri döndürür.
    İsteğe bağlı olarak CV metrikleriyle karşılaştırma tablosu gösterir ve MLflow'a loglar.

    Kullanım:
        test_metrics = evaluate_test(
            best_model, X_test, y_test, TASK_TYPE, best,
            cv_metrics=results[best]['val'],
            experiment_name=EXPERIMENT_NAME,
        )

    Returns:
        test_metrics: {metrik_adı: değer} dict
    """
    y_pred = model.predict(X_test)
    y_s    = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test
    title  = f"{model_name.upper()} — Test Seti" if model_name else "Test Seti"

    if task_type == "binary":
        y_pred_prob  = model.predict_proba(X_test)[:, 1]
        test_metrics = {
            "F2"       : fbeta_score(y_test, y_pred, beta=2),
            "PR-AUC"   : average_precision_score(y_test, y_pred_prob),
            "ROC-AUC"  : roc_auc_score(y_test, y_pred_prob),
            "F1"       : f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall"   : recall_score(y_test, y_pred),
        }
        print(classification_report(y_test, y_pred))

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred),
            display_labels=sorted(y_s.unique()),
        ).plot(ax=axes[0], colorbar=False, cmap="Blues")
        axes[0].set_title("Confusion Matrix")

        prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
        axes[1].plot(rec, prec, color="#2196F3", lw=2)
        axes[1].axhline(y_s.mean(), color="red", linestyle="--", alpha=0.6,
                        label=f"Baseline ({y_s.mean():.2f})")
        axes[1].set(xlabel="Recall", ylabel="Precision",
                    title=f'PR Curve — AUC={test_metrics["PR-AUC"]:.3f}')
        axes[1].legend()

        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        axes[2].plot(fpr, tpr, color="#4CAF50", lw=2)
        axes[2].plot([0, 1], [0, 1], "r--", alpha=0.6)
        axes[2].set(xlabel="FPR", ylabel="TPR",
                    title=f'ROC Curve — AUC={test_metrics["ROC-AUC"]:.3f}')

    elif task_type == "multiclass":
        y_pred_prob  = model.predict_proba(X_test)
        test_metrics = {
            "ROC-AUC"  : roc_auc_score(y_test, y_pred_prob, multi_class="ovr", average="weighted"),
            "F1"       : f1_score(y_test, y_pred, average="macro"),
            "Precision": precision_score(y_test, y_pred, average="macro"),
            "Recall"   : recall_score(y_test, y_pred, average="macro"),
        }
        print(classification_report(y_test, y_pred))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred),
            display_labels=sorted(y_s.unique()),
        ).plot(ax=axes[0], colorbar=False, cmap="Blues")
        axes[0].set_title("Confusion Matrix")

        ms = pd.Series({k: round(v, 3) for k, v in test_metrics.items()})
        axes[1].bar(ms.index, ms.values, color="#2196F3")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Test Metrikleri")
        for i, v in enumerate(ms.values):
            axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    elif task_type in ("regression", "timeseries"):
        test_metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAE" : float(mean_absolute_error(y_test, y_pred)),
            "R2"  : float(r2_score(y_test, y_pred)),
        }

        residuals = y_s.values - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set(xlabel="Tahmin", ylabel="Artık", title="Residual Plot")

        axes[1].hist(residuals, bins=40, color="#2196F3", edgecolor="white")
        axes[1].set_title(
            f'Artık Dağılımı  |  RMSE={test_metrics["RMSE"]:.3f}  R²={test_metrics["R2"]:.3f}'
        )

    elif task_type == "multilabel":
        test_metrics = {
            "F1"       : f1_score(y_test, y_pred, average="samples"),
            "Precision": precision_score(y_test, y_pred, average="samples"),
            "Recall"   : recall_score(y_test, y_pred, average="samples"),
        }

        ms = pd.Series({k: round(v, 3) for k, v in test_metrics.items()})
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.suptitle(title, fontsize=13, fontweight="bold")
        ax.bar(ms.index, ms.values, color="#2196F3")
        ax.set_ylim(0, 1)
        for i, v in enumerate(ms.values):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    else:
        raise ValueError(f"Desteklenmeyen task_type: '{task_type}'")

    print("\n=== Test Metrikleri ===")
    for k, v in test_metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    # CV vs Test karşılaştırma tablosu
    if cv_metrics is not None:
        cv_key_map = _CV_KEY_MAP.get(task_type, {})
        comparison = pd.DataFrame({
            "CV (val)": {k: round(cv_metrics.get(v, float("nan")), 3)
                         for k, v in cv_key_map.items()},
            "Test":     {k: round(test_metrics.get(k, float("nan")), 3)
                         for k in cv_key_map},
        })
        comparison["Fark"] = (comparison["Test"] - comparison["CV (val)"]).round(3)
        prefix = f"{model_name.upper()} — " if model_name else ""
        print(f"\n=== {prefix}CV vs Test Karşılaştırması ===")
        display(comparison)

    plt.tight_layout()

    # MLflow loglama
    if experiment_name:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            run_name = (
                f"test_eval_{model_name}_{datetime.now():%Y%m%d_%H%M}"
                if model_name else f"test_eval_{datetime.now():%Y%m%d_%H%M}"
            )
            with mlflow.start_run(run_name=run_name,
                                  experiment_id=experiment.experiment_id):
                tags = {
                    "type":  "test_evaluation",
                    "stage": "test",
                    "model": model_name,
                }
                if feature_version:
                    tags["feature_version"] = feature_version
                if extra_tags:
                    tags.update(extra_tags)
                mlflow.set_tags(tags)
                mlflow.log_metrics({
                    f"test_{k.lower().replace('-', '_')}": v
                    for k, v in test_metrics.items()
                })
                mlflow.log_figure(fig, "test_evaluation.png")
                if features is not None:
                    mlflow.log_dict({"features": features}, "features.json")
            print(f"✓ Test metrikleri MLflow'a kaydedildi")

    plt.show()

    return test_metrics
