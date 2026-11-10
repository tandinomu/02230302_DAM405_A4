"""Train stage: fit a scaled logistic-regression model and export it as a JSON artefact.

Why JSON and not a pickle?
  * The Lambda handler can score the model with pure Python, so the deployment
    package has no numpy/scipy/scikit-learn (kilobytes instead of ~200 MB).
    That keeps cold starts short and stays far below Lambda's 250 MB limit.
  * JSON is human-readable, diff-able and safe to load (unpickling is not).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CLASS_NAMES, FEATURES, TARGET, PipelineConfig
from .io_utils import write_json_atomic

log = logging.getLogger(__name__)


class ModelQualityError(Exception):
    """Raised when the trained model does not meet the minimum accuracy."""


def fit_pipeline(train_df: pd.DataFrame, random_state: int) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    model.fit(train_df[FEATURES], train_df[TARGET])
    return model


def export_model(model: Pipeline, run_id: str, metrics: dict) -> dict:
    scaler: StandardScaler = model.named_steps["scaler"]
    clf: LogisticRegression = model.named_steps["clf"]
    return {
        "model_version": f"wine-logreg-{run_id}",
        "algorithm": "StandardScaler + multinomial LogisticRegression",
        "features": FEATURES,
        "classes": [CLASS_NAMES[int(c)] for c in clf.classes_],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def train(paths: dict[str, str], run_id: str, cfg: PipelineConfig) -> dict[str, str]:
    train_df = pd.read_csv(paths["train"])
    test_df = pd.read_csv(paths["test"])

    model = fit_pipeline(train_df, cfg.random_state)
    preds = model.predict(test_df[FEATURES])
    metrics = {
        "accuracy": round(float(accuracy_score(test_df[TARGET], preds)), 4),
        "f1_macro": round(float(f1_score(test_df[TARGET], preds, average="macro")), 4),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    log.info("Evaluation metrics: %s", metrics)
    log.info("Classification report:\n%s",
             classification_report(test_df[TARGET], preds, target_names=CLASS_NAMES))

    out_dir = Path(cfg.run_dir("models", run_id))
    model_path = write_json_atomic(export_model(model, run_id, metrics), out_dir / "model.json")
    metrics_path = write_json_atomic(metrics, out_dir / "metrics.json")

    # Model-quality gate: never deploy a model that is worse than the agreed threshold.
    if metrics["accuracy"] < cfg.min_accuracy:
        raise ModelQualityError(
            f"accuracy {metrics['accuracy']:.3f} is below MIN_ACCURACY={cfg.min_accuracy:.2f}; not deploying"
        )
    return {"model": str(model_path), "metrics": str(metrics_path), "test": paths["test"]}
