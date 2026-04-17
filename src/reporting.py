from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow

from src.modeling import MAIN_METRIC


def _fmt(val) -> str:
    try:
        return f"{float(val):.3f}"
    except (TypeError, ValueError):
        return str(val) if val is not None else "—"


def generate_experiment_report(
    experiment_name: str,
    task_type: str,
    dataset_path: str,
    n_samples: int,
    hit_rate: float,
    n_features: int,
) -> str:
    """
    MLflow'daki tüm run'ları çekerek markdown raporu üretir,
    docs/ artifact olarak loglar ve rapor metnini döndürür.

    Kullanım:
        report = generate_experiment_report(
            experiment_name=EXPERIMENT_NAME,
            task_type=TASK_TYPE,
            dataset_path=DATASET_PATH,
            n_samples=len(df_model),
            hit_rate=y.mean(),
            n_features=len(FEATURES),
        )
        print(report)
    """
    main_metric = MAIN_METRIC[task_type]

    all_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["start_time DESC"],
    )

    def _filter(tag_val):
        if "tags.type" not in all_runs.columns:
            return all_runs.iloc[0:0]
        return all_runs[all_runs["tags.type"] == tag_val]

    # Her model tipi için sadece en güncel run'ı göster
    baseline_runs = (
        _filter("baseline")
        .drop_duplicates(subset=["tags.model_type"], keep="first")
    )
    tuning_runs = (
        _filter("tuning")
        .drop_duplicates(subset=["tags.model_type"], keep="first")
    )
    test_runs = _filter("test_evaluation").head(1)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Experiment Raporu — {experiment_name}",
        f"",
        f"**Oluşturulma:** {now_str}  ",
        f"**Task:** {task_type}  ",
        f"**Ana Metrik:** {main_metric}  ",
        f"**Dataset:** {dataset_path}  ",
        f"**Toplam Film:** {n_samples:,}  |  **Hit Oranı:** %{hit_rate * 100:.1f}  ",
        f"**Özellik Sayısı:** {n_features}  ",
        f"",
        f"---",
        f"",
        f"## Baseline Sonuçları",
        f"",
        f"| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |",
        f"|-------|--------|---------|----|-----------|----|",
    ]

    metric_cols = {
        "PR-AUC":    "metrics.avg_prec",
        "ROC-AUC":   "metrics.roc_auc",
        "F1":        "metrics.f1",
        "Precision": "metrics.precision",
        "Recall":    "metrics.recall",
    }

    for _, row in baseline_runs.iterrows():
        model = row.get("tags.model_type", "—")
        vals  = [_fmt(row.get(col)) for col in metric_cols.values()]
        lines.append(f"| {model} | " + " | ".join(vals) + " |")

    lines += ["", "---", "", "## Tuning", ""]

    for _, row in tuning_runs.iterrows():
        model    = row.get("tags.model_type", "—")
        baseline = _fmt(row.get(f"metrics.baseline_{main_metric}"))
        tuned    = _fmt(row.get(f"metrics.tuned_{main_metric}"))
        imp      = _fmt(row.get("metrics.improvement"))
        lines.append(
            f"- **{model}** — Baseline {main_metric}: {baseline}"
            f" → Tuned: {tuned} (İyileşme: {imp})"
        )

    lines += ["", "---", "", "## Test Seti Sonuçları", "", "| Metrik | Değer |", "|--------|-------|"]

    for _, row in test_runs.iterrows():
        for col in all_runs.columns:
            if col.startswith("metrics.test_"):
                metric_name = col.replace("metrics.test_", "").upper()
                lines.append(f"| {metric_name} | {_fmt(row.get(col))} |")
        break  # sadece ilk test run'ı

    report = "\n".join(lines)

    # Timestamp'li dosya adıyla MLflow'a artifact olarak kaydet
    report_filename = f"experiment_report_{datetime.now():%Y%m%d_%H%M%S}.md"
    tmp_path = Path(tempfile.gettempdir()) / report_filename
    tmp_path.write_text(report, encoding="utf-8")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    with mlflow.start_run(
        run_name=f"report_{datetime.now():%Y%m%d_%H%M}",
        experiment_id=experiment.experiment_id,
    ):
        mlflow.set_tag("type", "report")
        mlflow.log_artifact(str(tmp_path), artifact_path="docs")
        print(f"✓ Rapor oluşturuldu → {report_filename}")
        print(f"✓ MLflow'a kaydedildi → docs/{report_filename}")

    tmp_path.unlink()
    return report
