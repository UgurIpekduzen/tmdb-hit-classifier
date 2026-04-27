from __future__ import annotations

import logging
import warnings
from datetime import datetime
from typing import Callable

logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="mlflow")

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling import MAIN_METRIC, RANDOM_STATE, get_cv

_SKLEARN_SCORING = {
    "avg_prec": "average_precision",
    "f1":       "f1_macro",
    "f2":       make_scorer(fbeta_score, beta=2),
    "rmse":     "neg_root_mean_squared_error",
    "mae":      "neg_mean_absolute_error",
    "r2":       "r2",
}

_DIRECTION = {
    "avg_prec": "maximize",
    "f1":       "maximize",
    "f2":       "maximize",
    "rmse":     "minimize",
    "mae":      "minimize",
    "r2":       "maximize",
}


_GAP_PENALTY_FACTOR = 4.0   # Her 0.01 aşım için skoru 0.04 düşür/artır


# ── Yerleşik model factory'leri ────────────────────────────────────────────────

def _lgbm_fn(ir: int) -> Callable:
    """LGBMClassifier için varsayılan Optuna param uzayı."""
    def factory(trial):
        return LGBMClassifier(
            n_estimators      = trial.suggest_int("n_estimators", 100, 800),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 15, 63),
            min_child_samples = trial.suggest_int("min_child_samples", 20, 100),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 0.9),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 0.9),
            scale_pos_weight  = ir,
            random_state      = RANDOM_STATE,
            verbose           = -1,
            n_jobs            = 1,
        )
    return factory


def _rf_fn(_ir: int) -> Callable:
    """RandomForestClassifier için varsayılan Optuna param uzayı."""
    def factory(trial):
        return RandomForestClassifier(
            n_estimators     = trial.suggest_int("n_estimators", 100, 500),
            max_depth        = trial.suggest_int("max_depth", 5, 20),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 30),
            max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            class_weight     = "balanced",
            random_state     = RANDOM_STATE,
            n_jobs           = 1,
        )
    return factory


def _logreg_fn(_ir: int) -> Callable:
    """LogisticRegression (Pipeline) için varsayılan Optuna param uzayı."""
    def factory(trial):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C            = trial.suggest_float("C", 1e-3, 10, log=True),
                class_weight = "balanced",
                max_iter     = 1000,
                random_state = RANDOM_STATE,
            )),
        ])
    return factory


_BUILTIN_MODEL_FN: dict[str, Callable] = {
    "lgbm":   _lgbm_fn,
    "rf":     _rf_fn,
    "logreg": _logreg_fn,
}


# ── Objective builder ──────────────────────────────────────────────────────────

def _build_objective(model_fn: Callable, X, y, task_type: str,
                     max_gap: float | None = None):
    """
    model_fn : trial → sklearn estimator
        Her trial'da bir model örneği oluşturur.
        Yerleşik string tabanlı çağrı (lgbm/rf/logreg) veya kullanıcı tanımlı callable.
    """
    main_metric     = MAIN_METRIC[task_type]
    sklearn_scoring = _SKLEARN_SCORING[main_metric]
    cv              = get_cv(task_type)
    is_minimize     = main_metric in ("rmse", "mae")

    def objective(trial):
        model = model_fn(trial)

        if max_gap is not None:
            # Train + val skorlarını birlikte hesapla
            raw         = cross_validate(model, X, y, cv=cv, scoring=sklearn_scoring,
                                         return_train_score=True, n_jobs=1)
            val_score   = abs(float(np.mean(raw["test_score"])))
            train_score = abs(float(np.mean(raw["train_score"])))
            gap         = abs(train_score - val_score)
            excess      = max(0.0, gap - max_gap)

            # Maximize metrikler için penalize et (düşür), minimize için artır
            value = val_score + excess * _GAP_PENALTY_FACTOR * (-1 if not is_minimize else 1)
            trial.set_user_attr("gap",        round(gap, 4))
            trial.set_user_attr("val_score",  round(val_score, 4))
        else:
            scores = cross_val_score(model, X, y, cv=cv, scoring=sklearn_scoring, n_jobs=1)
            value  = float(np.mean(scores))
            if is_minimize:
                value = abs(value)

        return value

    return objective


def tune_model(model_name: str, X, y, task_type: str,
               baseline_score: float, n_trials: int = 50,
               max_gap: float | None = None,
               model_fn: Callable | None = None,
               feature_version: str = ""):
    """
    Optuna ile hiperparametre araması yapar, MLflow'a loglar.

    Parameters
    ----------
    model_name : str
        Yerleşik modeller için: 'lgbm', 'rf', 'logreg'.
        model_fn verilmişse yalnızca loglama için kullanılır (herhangi bir string olabilir).
    model_fn : Callable | None
        ``trial → sklearn estimator`` imzasına sahip callable.
        Verilmezse model_name'e göre yerleşik factory seçilir.
        Herhangi bir sklearn-uyumlu model için gap penalizasyonunu etkinleştirir.
        Örnek::

            from xgboost import XGBClassifier

            def xgb_fn(trial):
                return XGBClassifier(
                    n_estimators  = trial.suggest_int("n_estimators", 100, 500),
                    max_depth     = trial.suggest_int("max_depth", 3, 10),
                    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    use_label_encoder=False, eval_metric="logloss",
                )

            best_xgb, study = tune_model(
                model_name="xgb", X=X_train, y=y_train,
                task_type=TASK_TYPE, baseline_score=0.80,
                model_fn=xgb_fn, max_gap=0.10,
            )
    max_gap : float | None
        Belirtilirse train-val gap bu eşiği aşan trial'lar penalize edilir.
        None (varsayılan) → yalnızca val skoru optimize edilir.

    Returns
    -------
    best_model : fit edilmiş en iyi model
    study      : Optuna study nesnesi
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    main_metric = MAIN_METRIC[task_type]
    direction   = _DIRECTION[main_metric]
    ir          = (
        (y == 0).sum() / (y == 1).sum()
        if task_type in ("binary", "multiclass") else 1
    )

    # model_fn belirleme: kullanıcı vermişse kullan, yoksa yerleşik factory'e bak
    if model_fn is None:
        if model_name not in _BUILTIN_MODEL_FN:
            raise ValueError(
                f"Bilinmeyen model_name: '{model_name}'. "
                f"Yerleşik modeller: {list(_BUILTIN_MODEL_FN)}. "
                "Özel model için model_fn parametresini kullanın."
            )
        model_fn = _BUILTIN_MODEL_FN[model_name](ir)

    gap_info = f"  |  max_gap: {max_gap}" if max_gap is not None else ""
    print(f"Optuna başlıyor — model: {model_name}  |  metrik: {main_metric}  |  trial: {n_trials}{gap_info}")

    study = optuna.create_study(direction=direction)
    study.optimize(
        _build_objective(model_fn, X, y, task_type, max_gap=max_gap),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    tuned_score = study.best_value
    improvement = tuned_score - baseline_score

    print(f"\n✓ Tuning tamamlandı")
    print(f"  Baseline  {main_metric}: {baseline_score:.4f}")
    print(f"  Tuned     {main_metric}: {tuned_score:.4f}")
    print(f"  İyileşme : {improvement:+.4f}")
    print(f"\n  En iyi parametreler:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # En iyi parametrelerle modeli yeniden oluştur (FixedTrial ile model_fn'i tekrar çağır)
    fixed_trial = optuna.trial.FixedTrial(study.best_params)
    best_model  = model_fn(fixed_trial)
    best_model.fit(X, y)

    # Gerçek val skoru (max_gap kullanıldıysa penalized score yerine)
    best_trial    = study.best_trial
    real_val_score = best_trial.user_attrs.get("val_score", tuned_score)
    best_gap       = best_trial.user_attrs.get("gap", None)

    if max_gap is not None and best_gap is not None:
        print(f"  Train-val gap: {best_gap:.4f}  ({'✓ eşik altı' if best_gap <= max_gap else '⚠ eşik üstü'})")

    # MLflow'a logla
    with mlflow.start_run(run_name=f"tuning_{model_name}_{datetime.now():%Y%m%d_%H%M}"):
        tags = {"stage": "tuning", "model_type": model_name, "type": "tuning"}
        if feature_version:
            tags["feature_version"] = feature_version
        mlflow.set_tags(tags)
        mlflow.log_params(study.best_params)
        mlflow.log_metric(f"tuned_{main_metric}",    real_val_score)
        mlflow.log_metric(f"baseline_{main_metric}", baseline_score)
        mlflow.log_metric("improvement", real_val_score - baseline_score)
        if max_gap is not None:
            mlflow.log_param("max_gap", max_gap)
        if best_gap is not None:
            mlflow.log_metric("best_gap", best_gap)

        estimator = best_model[-1] if isinstance(best_model, Pipeline) else best_model
        if isinstance(estimator, LGBMClassifier):
            mlflow.lightgbm.log_model(estimator, "model")
        else:
            mlflow.sklearn.log_model(best_model, "model")

        print("✓ Tuning sonuçları ve model MLflow'a kaydedildi")

    print(f"\n✓ best_model güncellendi — tuned {model_name}")
    return best_model, study
