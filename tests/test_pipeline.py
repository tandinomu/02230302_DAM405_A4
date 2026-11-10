import json
import os
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pytest
from moto import mock_aws

from pipeline.config import FEATURES
from pipeline.deploy import deploy
from pipeline.extract import extract
from pipeline.train import ModelQualityError, fit_pipeline, train
from pipeline.transform import transform
from pipeline.validate import DataQualityError, run_expectations, validate

RUN = "2026-11-06"


def test_quality_gate_passes_clean_data(cfg):
    raw = extract(RUN, cfg)
    assert validate(raw, RUN, cfg) == raw
    report = json.loads((cfg.data_dir / "quality" / RUN / "validation_report.json").read_text())
    assert report["success"] and report["failed"] == 0


def test_quality_gate_blocks_bad_data(cfg):
    raw = extract(RUN, cfg, inject_bad_rows=True)
    with pytest.raises(DataQualityError) as exc:
        validate(raw, RUN, cfg)
    for col in ("alcohol", "magnesium", "hue", "target"):
        assert col in str(exc.value)


def test_quality_gate_catches_schema_drift(cfg):
    df = pd.read_csv(extract(RUN, cfg)).drop(columns=["hue"])
    assert run_expectations(df)["success"] is False


def test_extract_and_transform_are_idempotent(cfg):
    raw1 = extract(RUN, cfg)
    first = {k: Path(v).read_bytes() for k, v in transform(raw1, RUN, cfg).items()}
    raw2 = extract(RUN, cfg)
    second = {k: Path(v).read_bytes() for k, v in transform(raw2, RUN, cfg).items()}
    assert raw1 == raw2 and first == second
    # atomic writes leave no temp files behind
    assert not list(cfg.data_dir.rglob("*.tmp"))


def test_exported_json_model_matches_sklearn(cfg, app):
    paths = transform(extract(RUN, cfg), RUN, cfg)
    artefacts = train(paths, RUN, cfg)
    model_json = json.loads(Path(artefacts["model"]).read_text())

    train_df, test_df = pd.read_csv(paths["train"]), pd.read_csv(paths["test"])
    sk = fit_pipeline(train_df, cfg.random_state)
    sk_proba = sk.predict_proba(test_df[FEATURES])

    for i, rec in enumerate(test_df[FEATURES].to_dict(orient="records")):
        pred = app.predict_one(model_json, rec)
        assert np.allclose(list(pred["probabilities"].values()), sk_proba[i], atol=1e-3)


def test_model_quality_gate(cfg, monkeypatch):
    monkeypatch.setenv("MIN_ACCURACY", "1.01")  # impossible threshold
    from pipeline.config import PipelineConfig
    strict = PipelineConfig()
    paths = transform(extract(RUN, strict), RUN, strict)
    with pytest.raises(ModelQualityError):
        train(paths, RUN, strict)


def test_local_deploy_smoke_test(cfg):
    artefacts = train(transform(extract(RUN, cfg), RUN, cfg), RUN, cfg)
    result = deploy(artefacts, RUN, cfg)
    assert result["matches"] == len(result["expected"])
    assert (cfg.data_dir / "artefacts" / "production" / "model.json").exists()


@mock_aws
def test_aws_deploy_uploads_model_to_s3(cfg, monkeypatch):
    """S3 side of DEPLOY_MODE=aws; the Lambda update/invoke is exercised against LocalStack or AWS."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    artefacts = train(transform(extract(RUN, cfg), RUN, cfg), RUN, cfg)
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="models")
    s3.upload_file(artefacts["model"], "models", f"models/wine/{RUN}/model.json")
    obj = s3.get_object(Bucket="models", Key=f"models/wine/{RUN}/model.json")
    assert json.loads(obj["Body"].read())["model_version"] == f"wine-logreg-{RUN}"
