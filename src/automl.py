from __future__ import annotations

import numpy as np

from src.modeling import MAIN_METRIC, _OVERFIT_THRESHOLD
from src.tuning import tune_model


def recover_overfit(
    results: dict,
    best_model,
    best_name: str,
    study,
    X,
    y,
    task_type: str,
    max_gap: float | None = None,
    n_trials: int = 25,
    model_fn_map: dict | None = None,
) -> tuple:
    """
    Baseline'da overfit olan modelleri recovery tuning ile tune eder.
    Tuned best model'den daha iyiyse best modeli günceller.

    Notebook'ta tune_model() çağrısından hemen sonra kullanılır:

        best_model, study    = tune_model(best, ...)
        best_model, best     = recover_overfit(results, best_model, best, study, ...)
        test_metrics         = evaluate_test(best_model, ...)

    Parameters
    ----------
    results : dict
        train_session() çıktısı — overfit tespiti için kullanılır.
    best_model :
        tune_model() ile elde edilmiş fit edilmiş model.
    best_name : str
        Mevcut best modelin adı.
    study :
        tune_model() çıktısı — tuned val skorunu çıkarmak için.
    max_gap : float | None
        Overfit tespiti ve penalizasyon eşiği.
        None → _OVERFIT_THRESHOLD (0.20) kullanılır.
    n_trials : int
        Recovery tuning trial sayısı.
    model_fn_map : dict | None
        Yerleşik olmayan modeller için {model_name: model_fn} eşlemesi.

    Returns
    -------
    best_model : güncellenmiş (veya değişmemiş) en iyi model
    best_name  : güncellenmiş (veya değişmemiş) en iyi model adı
    """
    main_metric   = MAIN_METRIC[task_type]
    gap_threshold = max_gap if max_gap is not None else _OVERFIT_THRESHOLD
    best_score    = study.best_trial.user_attrs.get("val_score", study.best_value)

    overfit = {
        n: r for n, r in results.items()
        if r["train"][main_metric] - r["val"][main_metric] > gap_threshold
    }

    if not overfit:
        print("✓ Overfit model yok — recovery atlandı.")
        return best_model, best_name

    print(f"\n⚠️  Recovery tuning: {list(overfit.keys())}  (eşik={gap_threshold})")

    for name, r in overfit.items():
        fn = (model_fn_map or {}).get(name)

        print(f"\n── Recovery Tuning: {name} ──")
        rec_model, rec_study = tune_model(
            model_name     = name,
            X              = X,
            y              = y,
            task_type      = task_type,
            baseline_score = r["val"][main_metric],
            n_trials       = n_trials,
            max_gap        = max_gap,
            model_fn       = fn,
        )
        rec_score = rec_study.best_trial.user_attrs.get("val_score", rec_study.best_value)
        rec_gap   = rec_study.best_trial.user_attrs.get("gap", None)

        if rec_gap is not None and rec_gap > gap_threshold:
            print(f"  ✗ {name}: recovery sonrası hala overfit (gap={rec_gap:.4f}) — elendi")
            continue

        if rec_score > best_score:
            print(f"  ✓ {name}: recovery başarılı ({rec_score:.4f} > {best_score:.4f}) — best model güncellendi")
            best_model = rec_model
            best_score = rec_score
            best_name  = name
        else:
            print(f"  — {name}: recovery tamamlandı ({rec_score:.4f} ≤ {best_score:.4f}) — best model değişmedi")

    return best_model, best_name


def tune_all_models(
    results: dict,
    X,
    y,
    task_type: str,
    n_trials: int = 50,
    max_gap: float | None = None,
    model_fn_map: dict | None = None,
) -> tuple:
    """
    Tüm modelleri tune eder; geçerli modeller normal tuning, overfit modeller
    recovery tuning alır. En iyi tuned modeli döndürür.

    min_improvement her model için CV standart sapmasından otomatik türetilir —
    kullanıcı girdisi gerekmez.

    Parameters
    ----------
    results : dict
        train_session() çıktısı. Her entry'de "val", "train", "val_std" bulunmalıdır.
    n_trials : int
        Her model için Optuna trial sayısı.
    max_gap : float | None
        Overfit tespiti ve penalizasyon eşiği (None → _OVERFIT_THRESHOLD kullanılır).
    model_fn_map : dict | None
        Yerleşik olmayan modeller için {model_name: model_fn} eşlemesi.

    Returns
    -------
    best_model : fit edilmiş en iyi model
    best_name  : en iyi model adı
    """
    main_metric   = MAIN_METRIC[task_type]
    gap_threshold = _OVERFIT_THRESHOLD

    overfit_models = {
        n: r for n, r in results.items()
        if r["train"][main_metric] - r["val"][main_metric] > gap_threshold
    }
    valid_models = {n: r for n, r in results.items() if n not in overfit_models}

    best_model = None
    best_name  = None
    best_score = -np.inf

    # 1. Geçerli modelleri tune et
    for name, r in valid_models.items():
        baseline_score  = r["val"][main_metric]
        min_improvement = r.get("val_std", {}).get(main_metric, 0.01)

        print(f"\n── Tuning: {name}  (baseline={baseline_score:.4f}  min_improvement={min_improvement:.4f}) ──")
        fn = (model_fn_map or {}).get(name)
        tuned_model, study = tune_model(
            model_name     = name,
            X              = X,
            y              = y,
            task_type      = task_type,
            baseline_score = baseline_score,
            n_trials       = n_trials,
            max_gap        = max_gap,
            model_fn       = fn,
        )
        tuned_score = study.best_trial.user_attrs.get("val_score", study.best_value)
        improvement = tuned_score - baseline_score

        if improvement < min_improvement:
            print(f"  — {name}: iyileşme yetersiz ({improvement:+.4f} < {min_improvement:.4f}) — elendi")
            if tuned_score > best_score:
                best_model = tuned_model
                best_name  = name
                best_score = tuned_score
            continue

        if tuned_score > best_score:
            best_model = tuned_model
            best_name  = name
            best_score = tuned_score
            print(f"  ✓ {name}: aday  ({tuned_score:.4f})")

    # 2. Overfit modelleri recovery tune et
    if overfit_models:
        print(f"\n⚠️  Recovery tuning: {list(overfit_models.keys())}  (eşik={gap_threshold})")
        for name, r in overfit_models.items():
            fn = (model_fn_map or {}).get(name)
            print(f"\n── Recovery Tuning: {name} ──")
            rec_model, rec_study = tune_model(
                model_name     = name,
                X              = X,
                y              = y,
                task_type      = task_type,
                baseline_score = r["val"][main_metric],
                n_trials       = n_trials,
                max_gap        = max_gap,
                model_fn       = fn,
            )
            rec_score = rec_study.best_trial.user_attrs.get("val_score", rec_study.best_value)
            rec_gap   = rec_study.best_trial.user_attrs.get("gap", None)

            if rec_gap is not None and rec_gap > gap_threshold:
                print(f"  ✗ {name}: recovery sonrası hala overfit (gap={rec_gap:.4f}) — elendi")
                continue

            if rec_score > best_score:
                print(f"  ✓ {name}: recovery başarılı ({rec_score:.4f} > {best_score:.4f}) — best model güncellendi")
                best_model = rec_model
                best_name  = name
                best_score = rec_score
            else:
                print(f"  — {name}: recovery tamamlandı ({rec_score:.4f} ≤ {best_score:.4f}) — best model değişmedi")

    if best_model is None:
        raise RuntimeError("Hiçbir model tune edilemedi — pipeline durduruluyor.")

    print(f"\n✓ tune_all_models tamamlandı — best: {best_name}  ({main_metric}={best_score:.4f})")
    return best_model, best_name

    return best_model, best_name
