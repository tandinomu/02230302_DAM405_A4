"""Deploy stage: publish the approved model and smoke-test the live endpoint.

DEPLOY_MODE=aws   (real AWS or LocalStack via AWS_ENDPOINT_URL)
    1. upload model.json to s3://MODEL_BUCKET/models/wine/<run_id>/model.json
    2. point the Lambda function at the new key (MODEL_S3_KEY env var)
    3. wait for the update, invoke the function with a held-out sample, check it
DEPLOY_MODE=local
    copy the model into artefacts/production/ and smoke-test the handler in-process
    (useful when neither AWS nor LocalStack is available).

Re-running for the same run_id writes the same S3 key and the same env value, so
the task is idempotent.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from .config import CLASS_NAMES, FEATURES, REPO_ROOT, TARGET, PipelineConfig
from .io_utils import read_json, write_json_atomic

log = logging.getLogger(__name__)
HANDLER_PATH = REPO_ROOT / "src" / "inference" / "app.py"


class SmokeTestError(Exception):
    pass


def _sample_request(test_csv: str, n: int = 3) -> tuple[dict, list[str]]:
    df = pd.read_csv(test_csv).head(n)
    instances = df[FEATURES].to_dict(orient="records")
    expected = [CLASS_NAMES[int(t)] for t in df[TARGET]]
    return {"instances": instances}, expected


def _check(body: dict, expected: list[str], model_version: str) -> dict:
    if body.get("model_version") != model_version:
        raise SmokeTestError(f"endpoint serves {body.get('model_version')}, expected {model_version}")
    got = [p["label"] for p in body["predictions"]]
    log.info("Smoke test predictions=%s expected=%s", got, expected)
    return {"predictions": got, "expected": expected, "matches": sum(a == b for a, b in zip(got, expected))}


def _deploy_local(model_path: str, model: dict, request: dict, expected: list[str], cfg: PipelineConfig) -> dict:
    prod_dir = cfg.data_dir / "artefacts" / "production"
    prod_dir.mkdir(parents=True, exist_ok=True)
    tmp = prod_dir / ".model.json.tmp"
    shutil.copyfile(model_path, tmp)
    tmp.replace(prod_dir / "model.json")

    spec = importlib.util.spec_from_file_location("inference_app", HANDLER_PATH)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    app._MODEL = {**model, "_source": str(prod_dir / "model.json")}
    resp = app.lambda_handler(request, None)
    if resp["statusCode"] != 200:
        raise SmokeTestError(f"handler returned {resp['statusCode']}: {resp['body']}")
    return {"target": str(prod_dir / "model.json"), **_check(json.loads(resp["body"]), expected, model["model_version"])}


def _deploy_aws(model_path: str, model: dict, request: dict, expected: list[str],
                run_id: str, cfg: PipelineConfig) -> dict:
    import boto3

    if not (cfg.model_bucket and cfg.lambda_function_name):
        raise ValueError("DEPLOY_MODE=aws needs MODEL_BUCKET and LAMBDA_FUNCTION_NAME")
    session = boto3.session.Session(region_name=cfg.aws_region)
    s3 = session.client("s3", endpoint_url=cfg.aws_endpoint_url)
    lam = session.client("lambda", endpoint_url=cfg.aws_endpoint_url)

    key = f"models/wine/{run_id}/model.json"
    s3.upload_file(model_path, cfg.model_bucket, key, ExtraArgs={"ContentType": "application/json"})
    log.info("Uploaded model to s3://%s/%s", cfg.model_bucket, key)

    # Merge with the existing environment so we do not wipe other variables.
    current = lam.get_function_configuration(FunctionName=cfg.lambda_function_name)
    env = (current.get("Environment") or {}).get("Variables", {})
    env.update({"MODEL_S3_BUCKET": cfg.model_bucket, "MODEL_S3_KEY": key})
    lam.update_function_configuration(FunctionName=cfg.lambda_function_name, Environment={"Variables": env})
    lam.get_waiter("function_updated_v2").wait(FunctionName=cfg.lambda_function_name)
    log.info("Lambda %s now points at %s", cfg.lambda_function_name, key)

    resp = lam.invoke(FunctionName=cfg.lambda_function_name, Payload=json.dumps(request).encode())
    payload = json.loads(resp["Payload"].read())
    if resp.get("FunctionError") or payload.get("statusCode") != 200:
        raise SmokeTestError(f"smoke test failed: {payload}")
    return {"target": f"s3://{cfg.model_bucket}/{key}", "function": cfg.lambda_function_name,
            **_check(json.loads(payload["body"]), expected, model["model_version"])}


def deploy(artefacts: dict[str, str], run_id: str, cfg: PipelineConfig) -> dict:
    model = read_json(artefacts["model"])
    request, expected = _sample_request(artefacts["test"])
    if cfg.deploy_mode == "aws":
        result = _deploy_aws(artefacts["model"], model, request, expected, run_id, cfg)
    elif cfg.deploy_mode == "local":
        result = _deploy_local(artefacts["model"], model, request, expected, cfg)
    else:
        raise ValueError(f"unknown DEPLOY_MODE {cfg.deploy_mode!r}")

    result = {"run_id": run_id, "mode": cfg.deploy_mode, "model_version": model["model_version"], **result}
    write_json_atomic(result, Path(cfg.run_dir("deployments", run_id)) / "deployment.json")
    log.info("Deployment record: %s", result)
    return result
