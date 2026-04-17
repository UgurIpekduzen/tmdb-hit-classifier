from __future__ import annotations

import logging
import warnings
from pathlib import Path

logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="mlflow")

import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from IPython.display import display
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

VALID_TASK_TYPES = {"binary", "multiclass", "regression", "timeseries", "multilabel"}

MAIN_METRIC = {
    "binary":     "avg_prec",
    "multiclass": "f1",
    "regression": "rmse",
    "timeseries": "rmse",
    "multilabel": "f1",
}


def validate_task_type(task_type: str) -> None:
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f'Geçersiz task_type: "{task_type}". '
            f"Geçerli değerler: {VALID_TASK_TYPES}"
        )


def get_cv(task_type: str, n_splits: int = 5):
    if task_type in ("binary", "multiclass"):
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    elif task_type == "regression":
        return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    elif task_type == "timeseries":
        return TimeSeriesSplit(n_splits=n_splits)
    elif task_type == "multilabel":
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        return MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def get_scoring(task_type: str) -> dict:
    if task_type == "binary":
        return {
            "roc_auc":   "roc_auc",
            "f1":        "f1",
            "precision": "precision",
            "recall":    "recall",
            "avg_prec":  "average_precision",
        }
    elif task_type == "multiclass":
        return {
            "roc_auc":   "roc_auc_ovr_weighted",
            "f1":        "f1_macro",
            "precision": "precision_macro",
            "recall":    "recall_macro",
        }
    elif task_type in ("regression", "timeseries"):
        return {
            "rmse": "neg_root_mean_squared_error",
            "mae":  "neg_mean_absolute_error",
            "r2":   "r2",
        }
    elif task_type == "multilabel":
        return {
            "f1":        "f1_samples",
            "precision": "precision_samples",
            "recall":    "recall_samples",
        }


def compute_cv_metrics(model, X, y, cv, task_type: str) -> tuple[dict, dict, dict]:
    """CV ile metrikleri hesapla; (val_metrics, train_metrics, val_stds) döndürür."""
    scoring = get_scoring(task_type)
    raw = cross_validate(model, X, y, cv=cv, scoring=scoring,
                         return_train_score=True, n_jobs=1)

    def _extract(prefix, agg=np.mean):
        m = {k: float(agg(v)) for k, v in raw.items() if k.startswith(prefix)}
        m = {k.replace(prefix, ""): v for k, v in m.items()}
        m = {k: abs(v) if k in ("rmse", "mae") else v for k, v in m.items()}
        return m

    return _extract("test_"), _extract("train_"), _extract("test_", agg=np.std)


def _plot_residuals(model, X, y, title: str):
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Tahmin")
    axes[0].set_ylabel("Artık")
    axes[0].set_title("Residual Plot")
    axes[1].hist(residuals, bins=40, color="#2196F3", edgecolor="white")
    axes[1].set_title("Artık Dağılımı")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(model, X, y, title: str, task_type: str):
    """Sınıflandırma için confusion matrix, regresyon için residual plot döndürür."""
    if task_type in ("regression", "timeseries"):
        return _plot_residuals(model, X, y, title)

    model.fit(X, y)
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    labels = sorted(pd.Series(y).unique().tolist())
    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(3, len(labels))))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return fig


def _extract_shap_values(shap_values, task_type: str):
    """Ham SHAP çıktısından 2D (n_samples, n_features) array döndürür."""
    if task_type == "binary":
        if isinstance(shap_values, list):
            return shap_values[1]
        if shap_values.ndim == 3:
            return shap_values[:, :, 1]
        return shap_values
    elif task_type == "multiclass":
        if isinstance(shap_values, list):
            return np.mean([np.abs(sv) for sv in shap_values], axis=0)
        if shap_values.ndim == 3:
            return np.abs(shap_values).mean(axis=2)
        return shap_values
    else:
        if isinstance(shap_values, list):
            return shap_values[0]
        return shap_values


def log_shap(model, X, model_name: str = "", task_type: str = "binary") -> pd.Series:
    """SHAP summary/bar plot loglar; mean |SHAP| serisini döndürür."""
    estimator = model[-1] if isinstance(model, Pipeline) else model
    X_transformed = model[:-1].transform(X) if isinstance(model, Pipeline) else X
    feature_names = X.columns.tolist()

    if isinstance(estimator, (LGBMClassifier, RandomForestClassifier)):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_transformed)
    else:
        explainer = shap.LinearExplainer(estimator, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    sv = _extract_shap_values(shap_values, task_type)
    prefix = f"{model_name.upper()} — " if model_name else ""

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(sv, X_transformed, feature_names=feature_names,
                      show=False, max_display=20)
    ax.set_title(f"{prefix}SHAP Summary (Top 20)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if mlflow.active_run():
        mlflow.log_figure(fig, "shap_summary.png")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(sv, X_transformed, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20)
    ax.set_title(f"{prefix}SHAP Feature Importance (Top 20)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if mlflow.active_run():
        mlflow.log_figure(fig, "shap_importance.png")
    plt.show()
    plt.close(fig)

    mean_shap = pd.Series(np.abs(sv).mean(axis=0), index=feature_names)
    if mlflow.active_run():
        mlflow.log_metrics({f"shap_{col}": float(val) for col, val in mean_shap.items()})

    return mean_shap


def log_run(scenario: str, model_name: str, model, X, y, cv, task_type: str,
            extra_params: dict | None = None) -> dict:
    """Tek bir model run'ını nested olarak loglar; {val, train} metrik dict döndürür."""
    from datetime import datetime

    run_name = f"baseline_{model_name}_{datetime.now():%Y%m%d_%H%M}"

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tags({
            "stage":      "baseline",
            "scenario":   scenario,
            "model_type": model_name,
            "type":       "baseline",
        })

        estimator = model[-1] if isinstance(model, Pipeline) else model
        params = dict(estimator.get_params())
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        val_metrics, train_metrics, val_stds = compute_cv_metrics(model, X, y, cv, task_type)
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

        fig = plot_confusion_matrix(model, X, y, f"{scenario} — {model_name}", task_type)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        log_shap(model, X, model_name, task_type)

        if isinstance(estimator, LGBMClassifier):
            mlflow.lightgbm.log_model(estimator, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        main_metric = MAIN_METRIC[task_type]
        gap = train_metrics[main_metric] - val_metrics[main_metric]
        print(
            f"  ✓ {run_name:<50} "
            f"train={train_metrics[main_metric]:.3f}  "
            f"val={val_metrics[main_metric]:.3f}  "
            f"gap={gap:+.3f}"
        )

    return {"val": val_metrics, "train": train_metrics, "val_std": val_stds}


_OVERFIT_THRESHOLD  = 0.20
_UNDERFIT_THRESHOLD = 0.50


def compare_models(
    results: dict,
    task_type: str,
    experiment_name: str,
    y=None,
) -> str:
    """
    Model karşılaştırma tablosu, bar chart ve overfitting analizi yapar.
    Karşılaştırma grafiğini MLflow'a loglar.

    Kullanım:
        best = compare_models(results, TASK_TYPE, EXPERIMENT_NAME, y_train)

    Returns:
        best: En iyi model adı
    """
    main_metric = MAIN_METRIC[task_type]
    best = max(results, key=lambda n: results[n]["val"][main_metric])

    # 1. Metrik tablosu
    rows = []
    for name, m in results.items():
        row = {"Model": name}
        row.update({k.upper(): round(v, 3) for k, v in m["val"].items()})
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("Model")
    print("=== Model Karşılaştırma Tablosu ===")
    display(summary)
    print(f"\n🏆 En iyi model: {best}  ({main_metric}={results[best]['val'][main_metric]:.3f})")

    # 2. Bar chart — main metric karşılaştırması
    names       = list(results.keys())
    metric_vals = [results[n]["val"][main_metric] for n in names]
    colors      = ["#2196F3" if n == best else "#90CAF9" for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, metric_vals, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_title(f"Model Karşılaştırması — {main_metric.upper()}", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(metric_vals) * 1.15)
    ax.set_ylabel(main_metric.upper())
    if task_type in ("binary", "multiclass") and y is not None:
        y_mean = pd.Series(y).mean()
        ax.axhline(y=y_mean, color="red", linestyle="--", alpha=0.5,
                   label=f"Baseline ({y_mean:.2f})")
        ax.legend(fontsize=9)
    plt.tight_layout()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        with mlflow.start_run(run_name="comparison_chart",
                              experiment_id=experiment.experiment_id):
            mlflow.set_tag("type", "summary")
            mlflow.log_figure(fig, "model_comparison.png")
    plt.show()
    plt.close(fig)

    # 3. Overfitting analizi tablosu
    rows2 = []
    for name, m in results.items():
        train_s = m["train"][main_metric]
        val_s   = m["val"][main_metric]
        rows2.append({
            "Model":                         name,
            f"Train {main_metric.upper()}":  round(train_s, 3),
            f"Val {main_metric.upper()}":    round(val_s, 3),
            "Gap (Train−Val)":               round(train_s - val_s, 3),
        })
    gap_df = pd.DataFrame(rows2).set_index("Model")
    print("\n=== Overfitting / Underfitting Analizi ===")
    display(gap_df)

    # Grouped bar chart
    x          = np.arange(len(names))
    w          = 0.35
    train_vals = [results[n]["train"][main_metric] for n in names]
    val_vals   = [results[n]["val"][main_metric]   for n in names]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    b1 = ax2.bar(x - w / 2, train_vals, w, label="Train", color="#FF7043", alpha=0.85)
    b2 = ax2.bar(x + w / 2, val_vals,   w, label="Val",   color="#2196F3", alpha=0.85)
    ax2.bar_label(b1, fmt="%.3f", padding=3, fontsize=9)
    ax2.bar_label(b2, fmt="%.3f", padding=3, fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_title(f"Train vs Validation — {main_metric.upper()}", fontsize=12, fontweight="bold")
    ax2.set_ylabel(main_metric.upper())
    ax2.set_ylim(0, max(train_vals) * 1.2)
    ax2.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig2)

    # 4. Eşik kontrolü — overfit/underfit eden modeller uyarıyla dışlanır
    disqualified: set[str] = set()
    for name, m in results.items():
        gap       = m["train"][main_metric] - m["val"][main_metric]
        val_score = m["val"][main_metric]
        if gap > _OVERFIT_THRESHOLD:
            print(f"  ⚠️  {name}: ciddi overfit (gap={gap:.3f} > {_OVERFIT_THRESHOLD}) — dışlandı")
            disqualified.add(name)
        elif val_score < _UNDERFIT_THRESHOLD:
            print(f"  ⚠️  {name}: ciddi underfit (val={val_score:.3f} < {_UNDERFIT_THRESHOLD}) — dışlandı")
            disqualified.add(name)

    eligible = {n: m for n, m in results.items() if n not in disqualified}
    if not eligible:
        raise RuntimeError("Tüm modeller eşik kontrolünden başarısız oldu — pipeline durduruluyor.")

    best = max(eligible, key=lambda n: eligible[n]["val"][main_metric])

    if disqualified:
        print(f"  ✓ Geçerli modeller: {list(eligible.keys())} — en iyi: {best}")
    else:
        print("Tüm modeller eşik kontrolünden geçti — pipeline devam ediyor.")

    return best


def train_session(dataset_path: str, model_configs: dict, X, y,
                  task_type: str = "binary", scenario: str = "A") -> tuple[dict, dict]:
    """
    Tüm modelleri tek bir MLflow parent run altında eğitir.

    Kullanım:
        models, results = train_session(
            'data/tmdb_model.csv', MODEL_CONFIGS,
            X_train, y_train, task_type='binary'
        )
    """
    from datetime import datetime

    validate_task_type(task_type)
    cv = get_cv(task_type)
    imbalance_ratio = (
        int((y == 0).sum() / (y == 1).sum())
        if task_type in ("binary", "multiclass") else 1
    )

    experiment_name = Path(dataset_path).stem
    mlflow.set_experiment(experiment_name)

    session_name = f"session_{scenario}_{datetime.now():%Y%m%d_%H%M}"
    models = {name: factory(imbalance_ratio) for name, factory in model_configs.items()}

    print(
        f'Eğitim başlıyor  —  experiment: "{experiment_name}"'
        f"  |  oturum: {session_name}  |  task: {task_type}"
    )

    with mlflow.start_run(run_name=session_name):
        mlflow.set_tag("type", "session")
        mlflow.log_params({
            "dataset":    experiment_name,
            "n_train":    len(X),
            "n_features": X.shape[1],
            "task_type":  task_type,
            "scenario":   scenario,
            "models":     ",".join(model_configs.keys()),
        })

        results = {}
        for name, model in models.items():
            results[name] = log_run(scenario, name, model, X, y, cv, task_type)

    print("\n✓ Oturum tamamlandı")
    return models, results
