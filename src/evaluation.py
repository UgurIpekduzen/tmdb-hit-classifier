from __future__ import annotations

import logging
import warnings
from datetime import datetime

logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="mlflow")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
    f1_score, fbeta_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve,
    classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)

from src import mlflow_utils as mlu

MIN_SIG = 0.065  # Bootstrap CI minimum anlamlı Δ eşiği (Bölüm 12)

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

    # MLflow loglama (USE_MLFLOW=False ise atlanır)
    if experiment_name and mlu.USE_MLFLOW:
        import mlflow
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            run_name = (
                f"test_eval_{model_name}_{datetime.now():%Y%m%d_%H%M}"
                if model_name else f"test_eval_{datetime.now():%Y%m%d_%H%M}"
            )
            with mlu.maybe_run(run_name=run_name,
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
                mlu.set_tags(tags)
                mlu.log_metrics({
                    f"test_{k.lower().replace('-', '_')}": v
                    for k, v in test_metrics.items()
                })
                mlu.log_figure(fig, "test_evaluation.png")
                if features is not None:
                    mlu.log_dict({"features": features}, "features.json")
            print(f"✓ Test metrikleri MLflow'a kaydedildi")
    elif experiment_name:
        print("○ MLflow devre dışı (USE_MLFLOW=False) — test metrikleri kaydedilmedi")

    plt.show()

    return test_metrics


def register_champion(
    model,
    test_score: float,
    registry_name: str = "tmdb-hit-classifier",
    metric_name: str = "test_f2",
    feature_version: str = "",
) -> str:
    """
    Modeli MLflow Model Registry'ye kaydeder ve champion karşılaştırması yapar.

    Mevcut Production versiyonu varsa test_score ile karşılaştırır:
    - Challenger > Champion  → Production'a terfi, eski Production → Archived
    - Challenger ≤ Champion  → Staging'de bırakır

    Parameters
    ----------
    model        : fit edilmiş sklearn-uyumlu model
    test_score   : challenger'ın test metrik değeri (yüksek = iyi)
    registry_name: MLflow Model Registry'deki model adı
    metric_name  : karşılaştırma metriği (MLflow run'ında bu isimle loglanan)
    feature_version: etiket için feature seti versiyonu

    Returns
    -------
    stage : "Production" | "Staging" | "skipped"
    """
    if not mlu.USE_MLFLOW:
        print("○ MLflow devre dışı (USE_MLFLOW=False) — model registry'ye kayıt atlandı.")
        return "skipped"

    import mlflow
    import mlflow.lightgbm
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # Mevcut Production şampiyonu var mı?
    try:
        prod_versions = client.get_latest_versions(registry_name, stages=["Production"])
    except Exception:
        prod_versions = []

    champion_score = 0.0
    if prod_versions:
        try:
            run = client.get_run(prod_versions[0].run_id)
            champion_score = run.data.metrics.get(metric_name, 0.0)
        except Exception:
            pass

    # Challenger modeli yeni bir run'a logla
    run_name = f"registry_{datetime.now():%Y%m%d_%H%M}"
    with mlflow.start_run(run_name=run_name) as run:
        tags = {"stage": "registry", "registry_name": registry_name}
        if feature_version:
            tags["feature_version"] = feature_version
        mlflow.set_tags(tags)
        mlflow.log_metric(metric_name, test_score)

        estimator = model[-1] if isinstance(model, Pipeline) else model
        if isinstance(estimator, LGBMClassifier):
            mlflow.lightgbm.log_model(estimator, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        model_uri = f"runs:/{run.info.run_id}/model"

    # Registry'ye kaydet (Staging)
    mv = mlflow.register_model(model_uri, registry_name)
    client.transition_model_version_stage(
        name=registry_name, version=mv.version, stage="Staging"
    )

    # Champion karşılaştırması
    if test_score > champion_score:
        # Eski Production → Archived
        for v in prod_versions:
            client.transition_model_version_stage(
                name=registry_name, version=v.version, stage="Archived"
            )
        # Challenger → Production
        client.transition_model_version_stage(
            name=registry_name, version=mv.version, stage="Production"
        )
        stage = "Production"
        print(f"✓ Yeni şampiyon Production'a alındı  ({metric_name}: {test_score:.4f} > {champion_score:.4f})")
    else:
        stage = "Staging"
        print(f"  Challenger Staging'de kaldı  ({metric_name}: {test_score:.4f} ≤ {champion_score:.4f})")

    print(f"  Registry: {registry_name}  |  Version: {mv.version}  |  Stage: {stage}")
    return stage


def _bootstrap_f2_ci(
    y_test,
    y_pred,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Stratified bootstrap F2 %95 GA. (low, high) döndürür."""
    rng     = np.random.default_rng(seed)
    y_test  = np.array(y_test)
    y_pred  = np.array(y_pred)
    pos_idx = np.where(y_test == 1)[0]
    neg_idx = np.where(y_test == 0)[0]
    scores  = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        yt, yd = y_test[idx], y_pred[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        scores.append(fbeta_score(yt, yd, beta=2, zero_division=0))
    arr = np.array(scores)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def compare_feature_version(
    version: str,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    threshold: float,
    baseline_val_f2: float,
    baseline_test_f2: float,
    prev_cols: list[str] | None = None,
    drop_cols: list[str] | None = None,
    baseline_label: str = "prev",
    min_sig: float = MIN_SIG,
) -> tuple[float, float, list[str]]:
    """
    Clone model, optionally drop features, compare F2 vs baseline.

    prev_cols + drop_cols → new_cols. Eğer ikisi de None ise X_train tüm
    sütunlarla kullanılır (feature filtresi yok).

    Returns
    -------
    val_f2, test_f2, new_cols
    """
    if prev_cols is not None:
        drop = set(drop_cols or [])
        new_cols = [c for c in prev_cols if c not in drop]
        X_tr = X_train[new_cols]
        X_vl = X_val[new_cols]
        X_ts = X_test[new_cols]
        dropped = [c for c in (drop_cols or []) if c in prev_cols]
    else:
        new_cols = list(X_train.columns)
        X_tr, X_vl, X_ts = X_train, X_val, X_test
        dropped = []

    print(f"prev feature sayısı : {len(prev_cols) if prev_cols is not None else len(new_cols)}")
    print(f"{version} feature sayısı: {len(new_cols)}")
    if dropped:
        print(f"Drop edilenler      : {', '.join(dropped)}")

    clf = clone(model)
    clf.fit(X_tr, y_train)

    y_val_pred  = (clf.predict_proba(X_vl)[:, 1] >= threshold).astype(int)
    val_f2      = float(fbeta_score(np.array(y_val), y_val_pred, beta=2))

    y_test_pred = (clf.predict_proba(X_ts)[:, 1] >= threshold).astype(int)
    test_f2     = float(fbeta_score(np.array(y_test), y_test_pred, beta=2))
    ci_low, ci_high = _bootstrap_f2_ci(y_test, y_test_pred)

    prev_n   = len(prev_cols) if prev_cols is not None else len(new_cols)
    prev_lbl = f"{baseline_label} ({prev_n} feature)"
    new_lbl  = f"{version} ({len(new_cols)} feature)"
    hdr = f'{"Versiyon":<22s}  {"Val F2":>8s}  {"Test F2":>8s}  {"Δ Val":>8s}  {"Δ Test":>8s}  {"CI (test)":>22s}'
    print(f"\n=== {version} Feature Karşılaştırması (LGBM, clone test) ===")
    print(hdr)
    print("-" * 86)
    print(f'{prev_lbl:<22s}  {baseline_val_f2:>8.4f}  {baseline_test_f2:>8.4f}  {"−":>8s}  {"−":>8s}  {"−":>22s}')
    print(f'{new_lbl:<22s}  {val_f2:>8.4f}  {test_f2:>8.4f}  '
          f'{val_f2 - baseline_val_f2:>+8.4f}  {test_f2 - baseline_test_f2:>+8.4f}  '
          f'[{ci_low:.4f}, {ci_high:.4f}]')

    delta_val  = val_f2  - baseline_val_f2
    delta_test = test_f2 - baseline_test_f2
    sig        = abs(delta_test) >= min_sig

    print("\n--- Karar ---")
    if sig and delta_test > 0:
        print(f"  Δ Test F2 = {delta_test:+.4f} ≥ +{min_sig}  →  ✓ ANLAMLI İYİLEŞME")
        print(f"  → {version} kalıcı yapılabilir.")
    elif sig and delta_test < 0:
        print(f"  Δ Test F2 = {delta_test:+.4f} ≤ -{min_sig}  →  ✗ ANLAMLI BOZULMA — önceki versiyon korunur.")
    else:
        print(f"  Δ Test F2 = {delta_test:+.4f}  (|Δ| < {min_sig})  →  Gürültü sınırında.")
        print(f"  Δ Val  F2 = {delta_val:+.4f}")
        if dropped:
            print(f"  → Drop edilen feature'lar modeli bozmadı; {version} kalıcı yapılabilir.")
        else:
            print(f"  → {version} mevcut versiyonla istatistiksel olarak eşdeğer.")

    return val_f2, test_f2, new_cols


def bootstrap_ci(
    y_true,
    y_prob,
    y_pred=None,
    threshold: float = 0.5,
    metrics: list[str] | None = None,
    n_boot: int = 1000,
    ci_alpha: float = 0.05,
    beta: int = 2,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Binary sınıflandırma metrikleri için stratified bootstrap güven aralıkları.

    Parameters
    ----------
    y_true    : gerçek binary etiketler
    y_prob    : pozitif sınıf için tahmin edilen olasılıklar
    y_pred    : hard tahminler; None ise threshold'dan türetilir
    threshold : y_pred None olduğunda kullanılan karar eşiği (varsayılan 0.5)
    metrics   : ["F2","PR-AUC","ROC-AUC","Precision","Recall"] alt kümesi; None ise hepsi
    n_boot    : bootstrap iterasyon sayısı (varsayılan 1000)
    ci_alpha  : iki taraflı alpha seviyesi (varsayılan 0.05 → %95 GA)
    beta      : F-beta skoru için beta (varsayılan 2)
    seed      : tekrarlanabilirlik için random seed

    Returns
    -------
    Metrik bazlı indekslenmiş, point, ci_low, ci_high, ci_width sütunlarına sahip DataFrame

    Examples
    --------
    df_ci = bootstrap_ci(y_test, y_prob_test, threshold=OPTIMAL_THRESHOLD)
    """
    _fns = {
        "F2":        lambda yt, yp, yd: fbeta_score(yt, yd, beta=beta, zero_division=0),
        "PR-AUC":    lambda yt, yp, yd: average_precision_score(yt, yp),
        "ROC-AUC":   lambda yt, yp, yd: roc_auc_score(yt, yp),
        "Precision": lambda yt, yp, yd: precision_score(yt, yd, zero_division=0),
        "Recall":    lambda yt, yp, yd: recall_score(yt, yd, zero_division=0),
    }

    if metrics is None:
        metrics = list(_fns.keys())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int) if y_pred is None else np.array(y_pred)

    rng     = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    scores  = {m: [] for m in metrics}

    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        yp, yd = y_prob[idx], y_pred[idx]
        for m in metrics:
            scores[m].append(_fns[m](yt, yp, yd))

    lo_pct = 100 * ci_alpha / 2
    hi_pct = 100 * (1 - ci_alpha / 2)
    rows = []
    for m in metrics:
        arr = np.array(scores[m])
        pt  = float(_fns[m](y_true, y_prob, y_pred))
        lo  = float(np.percentile(arr, lo_pct))
        hi  = float(np.percentile(arr, hi_pct))
        rows.append({"metric": m, "point": round(pt, 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "ci_width": round(hi - lo, 4)})

    result = pd.DataFrame(rows).set_index("metric")
    if verbose:
        print(f"Bootstrap CI (n={len(y_true)}, B={n_boot}, {int((1-ci_alpha)*100)}% stratified)")
        print(result.to_string())
    return result


def feature_drop_test(
    df: pd.DataFrame,
    features: list[str],
    drop: list[str],
    target: str,
    model,
    mask_train,
    mask_val,
    mask_test,
    n_boot: int = 1000,
    sig_threshold: float = MIN_SIG,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Feature'ları drop etmenin istatistiksel olarak anlamlı bir F2 kaybına yol açıp açmadığını test eder.

    Model'in iki klonu eğitilir — biri tüm feature'larla, biri drop olmadan —
    val üzerinde optimal PR-curve eşiği bulunur, ardından test'te Bootstrap GA karşılaştırılır.

    Parameters
    ----------
    df            : tam DataFrame (feature'lar + target)
    features      : mevcut feature listesi (baseline)
    drop          : reduced model için çıkarılacak feature'lar
    target        : binary target sütunu
    model         : sklearn-uyumlu estimator (klonlanır, değiştirilmez)
    mask_train/val/test : temporal split'ler için boolean Series
    n_boot        : bootstrap iterasyon sayısı (varsayılan 1000)
    sig_threshold : anlamlı kabul edilecek minimum Δ (varsayılan MIN_SIG = 0.065)

    Returns
    -------
    ci_base : baseline model için bootstrap_ci DataFrame'i
    ci_drop : reduced model için bootstrap_ci DataFrame'i
    decision: 'drop' | 'keep'

    Examples
    --------
    from src.evaluation import feature_drop_test
    ci_base, ci_drop, decision = feature_drop_test(
        df, FEATURES, drop=['actor_hist_roi'], target='hit',
        model=tuned_lgbm, mask_train=mask_train, mask_val=mask_val, mask_test=mask_test,
    )
    """
    from src.tuning import optimal_threshold_from_pr

    feat_drop = [f for f in features if f not in drop]
    dropped   = [f for f in drop if f in features]

    print(f"Baseline : {len(features)} feature")
    print(f"Reduced  : {len(feat_drop)} feature  (drop: {', '.join(dropped)})")

    y_tr = df.loc[mask_train, target]
    y_vl = df.loc[mask_val,   target]
    y_te = df.loc[mask_test,  target]

    # ── Baseline ─────────────────────────────────────────────────────────────
    clf_base = clone(model)
    clf_base.fit(df.loc[mask_train, features], y_tr)
    thr_base, *_ = optimal_threshold_from_pr(
        y_vl, clf_base.predict_proba(df.loc[mask_val, features])[:, 1]
    )
    ci_base = bootstrap_ci(
        y_te, clf_base.predict_proba(df.loc[mask_test, features])[:, 1],
        threshold=thr_base, metrics=['F2', 'PR-AUC', 'ROC-AUC'],
        n_boot=n_boot, verbose=False,
    )

    # ── Reduced ──────────────────────────────────────────────────────────────
    clf_drop = clone(model)
    clf_drop.fit(df.loc[mask_train, feat_drop], y_tr)
    thr_drop, *_ = optimal_threshold_from_pr(
        y_vl, clf_drop.predict_proba(df.loc[mask_val, feat_drop])[:, 1]
    )
    ci_drop = bootstrap_ci(
        y_te, clf_drop.predict_proba(df.loc[mask_test, feat_drop])[:, 1],
        threshold=thr_drop, metrics=['F2', 'PR-AUC', 'ROC-AUC'],
        n_boot=n_boot, verbose=False,
    )

    # ── Report ───────────────────────────────────────────────────────────────
    print(f'\n{"Metrik":<10} {"Baseline":>10} {"Reduced":>10} {"Δ":>8}  Anlamlı?')
    print('-' * 50)
    for m in ['F2', 'PR-AUC', 'ROC-AUC']:
        base_v = ci_base.loc[m, 'point']
        drop_v = ci_drop.loc[m, 'point']
        delta  = drop_v - base_v
        sig    = '⚠️ Anlamlı' if abs(delta) >= sig_threshold else 'Gürültü'
        print(f'{m:<10} {base_v:>10.4f} {drop_v:>10.4f} {delta:>+8.4f}  {sig}')

    delta_f2 = ci_drop.loc['F2', 'point'] - ci_base.loc['F2', 'point']
    decision = 'drop' if abs(delta_f2) < sig_threshold else 'keep'
    label    = 'DROP ✅' if decision == 'drop' else 'KORU ❌'
    print(f'\nKarar: Δ F2 = {delta_f2:+.4f} — {label} (eşik={sig_threshold})')

    return ci_base, ci_drop, decision


def calibration_report(
    splits: list[tuple],
    model_name: str = "",
    n_bins: int = 10,
) -> None:
    """
    Bir veya daha fazla veri split'i için kalibrasyon eğrilerini ve Brier skorlarını çizer.

    Parameters
    ----------
    splits     : (split_name, y_true, y_prob) tuple'larından oluşan liste
    model_name : grafik başlığında kullanılan model tanımlayıcısı
    n_bins     : kalibrasyon bin sayısı (varsayılan 10)

    Examples
    --------
    calibration_report(
        [("Val", y_val, y_prob_val), ("Test", y_test, y_prob_test)],
        model_name="LightGBM",
    )
    """
    n = len(splits)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5))
    if n == 1:
        axes = [axes]
    title = f"Kalibrasyon Eğrisi — {model_name}" if model_name else "Kalibrasyon Eğrisi"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (split_name, y_true, y_prob) in zip(axes, splits):
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        brier = brier_score_loss(y_true, y_prob)
        label = f"{model_name}  (Brier={brier:.4f})" if model_name else f"Brier={brier:.4f}"
        ax.plot(mean_pred, frac_pos, "s-", color="#2196F3", lw=2, label=label)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Mükemmel Kalibrasyon")
        ax.set(xlabel="Tahmin Edilen Olasılık", ylabel="Gerçek Hit Oranı",
               title=f"Kalibrasyon Eğrisi — {split_name}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        print(f"  {split_name}: Brier={brier:.4f}  "
              f"(baseline={brier_score_loss(y_true, np.full(len(y_true), np.mean(y_true))):.4f})")

    plt.tight_layout()
    plt.show()
