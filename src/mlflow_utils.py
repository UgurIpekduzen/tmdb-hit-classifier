from __future__ import annotations

from contextlib import contextmanager

# Tek kontrol noktası: True olmadıkça hiçbir mlflow ağ çağrısı yapılmaz.
# Notebook config hücresi bunu `mlu.USE_MLFLOW = USE_MLFLOW` ile set eder.
USE_MLFLOW = False


@contextmanager
def maybe_run(*args, **kwargs):
    """mlflow.start_run() ile aynı arayüz; USE_MLFLOW=False iken no-op."""
    if not USE_MLFLOW:
        yield None
        return
    import mlflow
    with mlflow.start_run(*args, **kwargs) as run:
        yield run


def set_tags(tags: dict) -> None:
    if USE_MLFLOW:
        import mlflow
        mlflow.set_tags(tags)


def set_tag(key: str, value) -> None:
    if USE_MLFLOW:
        import mlflow
        mlflow.set_tag(key, value)


def log_params(params: dict) -> None:
    if USE_MLFLOW:
        import mlflow
        mlflow.log_params(params)


def log_param(key: str, value) -> None:
    if USE_MLFLOW:
        import mlflow
        mlflow.log_param(key, value)


def log_metrics(metrics: dict) -> None:
    if USE_MLFLOW:
        import mlflow
        mlflow.log_metrics(metrics)


def log_metric(key: str, value) -> None:
    if USE_MLFLOW:
        import mlflow
        mlflow.log_metric(key, value)


def log_figure(fig, path: str) -> None:
    if USE_MLFLOW:
        import mlflow
        mlflow.log_figure(fig, path)


def log_dict(d: dict, path: str) -> None:
    if USE_MLFLOW:
        import mlflow
        mlflow.log_dict(d, path)


def log_model(estimator, is_lgbm: bool) -> None:
    if USE_MLFLOW:
        import mlflow
        import mlflow.lightgbm
        import mlflow.sklearn
        if is_lgbm:
            mlflow.lightgbm.log_model(estimator, "model")
        else:
            mlflow.sklearn.log_model(estimator, "model")
