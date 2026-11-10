"""Central configuration for the pipeline.

Nothing secret lives here. Every value is read from an environment variable
(or an Airflow Variable passed in by the DAG) with a safe local default.
AWS credentials are NEVER read or stored by this code: boto3 resolves them
through its standard credential chain (env vars, ~/.aws profile, IAM role).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Feature order is part of the model contract shared by training and the Lambda handler.
FEATURES: list[str] = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline",
]
TARGET = "target"
CLASS_NAMES: list[str] = ["cultivar_0", "cultivar_1", "cultivar_2"]


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


@dataclass(frozen=True)
class PipelineConfig:
    # default_factory => values are read when the config is created (at task run time),
    # not when the module is imported, so env/Airflow changes are always picked up.
    data_dir: Path = field(default_factory=lambda: Path(_env("PIPELINE_DATA_DIR", str(REPO_ROOT / "data"))))
    min_accuracy: float = field(default_factory=lambda: float(_env("MIN_ACCURACY", "0.90")))
    random_state: int = field(default_factory=lambda: int(_env("RANDOM_STATE", "42")))
    test_size: float = field(default_factory=lambda: float(_env("TEST_SIZE", "0.25")))

    # Deployment
    deploy_mode: str = field(default_factory=lambda: _env("DEPLOY_MODE", "local"))  # "local" | "aws"
    model_bucket: str = field(default_factory=lambda: _env("MODEL_BUCKET", ""))
    lambda_function_name: str = field(default_factory=lambda: _env("LAMBDA_FUNCTION_NAME", ""))
    # Set to http://localhost:4566 to target LocalStack; leave empty for real AWS.
    aws_endpoint_url: str | None = field(default_factory=lambda: _env("AWS_ENDPOINT_URL", "") or None)
    aws_region: str = field(default_factory=lambda: _env("AWS_DEFAULT_REGION", "us-east-1"))

    def run_dir(self, stage: str, run_id: str) -> Path:
        """Partitioned output folder, e.g. data/raw/2026-11-06/ (one folder per logical run)."""
        path = self.data_dir / stage / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path
