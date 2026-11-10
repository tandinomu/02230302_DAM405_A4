"""Airflow DAG: extract -> data-quality gate -> transform -> train/evaluate -> deploy to Lambda.

Best-practice notes (DAG design)
  * Thin DAG, fat library: all logic lives in the `pipeline` package, which is
    unit-tested without Airflow. This file only wires tasks together.
  * Idempotent: every task writes to a folder partitioned by the run's logical
    date (data/<stage>/<YYYY-MM-DD>/) and overwrites it on re-run; training uses
    a fixed random_state; deployment writes a date-keyed S3 object.
  * Atomic: files are written to a temp file and renamed into place, so a failed
    task never leaves half-written output for the next task.
  * Small XComs: tasks pass file *paths*, never DataFrames.
  * No secrets in code: AWS credentials come from the standard boto3 chain
    (environment, ~/.aws profile or IAM role); settings come from env vars.
  * Gates fail fast: data-quality and model-quality failures raise
    AirflowFailException (no pointless retries) and block everything downstream.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

from airflow.exceptions import AirflowFailException

try:  # Airflow 3.x
    from airflow.sdk import Param, dag, get_current_context, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task
    from airflow.models.param import Param
    from airflow.operators.python import get_current_context

from pipeline.config import PipelineConfig

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "dam405",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


def _partition() -> str:
    """Folder key for this run: the logical date (YYYY-MM-DD), or a filesystem-safe run_id."""
    ctx = get_current_context()
    if ctx.get("ds"):
        return ctx["ds"]
    return re.sub(r"[^A-Za-z0-9_.-]", "-", ctx["dag_run"].run_id)


@dag(
    dag_id="wine_serverless_mlops_pipeline",
    description="Extract, validate (Great Expectations), transform, train and deploy a model to AWS Lambda",
    schedule="@daily",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["dam405", "mlops", "lambda", "great-expectations"],
    params={
        "source_csv": Param(None, type=["null", "string"],
                            description="Optional CSV path/URL. Empty = UCI Wine dataset bundled with scikit-learn."),
        "inject_bad_rows": Param(False, type="boolean",
                                 description="Append corrupted rows to demonstrate the data-quality gate failing."),
    },
    doc_md=__doc__,
)
def wine_serverless_mlops_pipeline():

    @task
    def extract() -> str:
        from pipeline.extract import extract as run_extract

        params = get_current_context()["params"]
        return run_extract(_partition(), PipelineConfig(),
                           source_csv=params.get("source_csv"),
                           inject_bad_rows=bool(params.get("inject_bad_rows")))

    @task(retries=0)
    def validate_data_quality(raw_path: str) -> str:
        from pipeline.validate import DataQualityError, validate

        try:
            return validate(raw_path, _partition(), PipelineConfig())
        except DataQualityError as exc:
            raise AirflowFailException(f"Data-quality gate FAILED: {exc}") from exc

    @task
    def transform(validated_path: str) -> dict:
        from pipeline.transform import transform as run_transform

        return run_transform(validated_path, _partition(), PipelineConfig())

    @task(retries=0)
    def train_and_evaluate(paths: dict) -> dict:
        from pipeline.train import ModelQualityError, train

        try:
            return train(paths, _partition(), PipelineConfig())
        except ModelQualityError as exc:
            raise AirflowFailException(f"Model-quality gate FAILED: {exc}") from exc

    @task
    def deploy_to_lambda(artefacts: dict) -> dict:
        from pipeline.deploy import deploy

        return deploy(artefacts, _partition(), PipelineConfig())

    deploy_to_lambda(train_and_evaluate(transform(validate_data_quality(extract()))))


wine_serverless_mlops_pipeline()
