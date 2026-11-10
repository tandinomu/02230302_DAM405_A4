"""AWS Lambda handler: serverless inference for the wine-cultivar classifier.

Supported triggers
  * API Gateway HTTP API (payload v2.0) -> POST /predict, GET /health
  * Lambda function URL (same v2.0 payload shape)
  * Direct invocation (`aws lambda invoke`, Step Functions, an Airflow task...)

Request body:
  {"instances": [{"alcohol": 13.2, "malic_acid": 1.78, ...}, ...]}
  or a single record: {"features": {"alcohol": 13.2, ...}}

Model loading
  The model is a small JSON file (scaler + logistic-regression weights). It is
  loaded ONCE per execution environment, at cold start, and cached in a module
  global, so warm invocations pay no loading cost. If MODEL_S3_BUCKET and
  MODEL_S3_KEY are set it is fetched from S3 (this is how the Airflow DAG rolls
  out a new model); otherwise the copy bundled in ./model/model.json is used.

The handler uses only the Python standard library (+ boto3, which the Lambda
runtime already provides), so the deployment package is a few kilobytes.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger()
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

BUNDLED_MODEL_PATH = Path(__file__).parent / "model" / "model.json"
MAX_INSTANCES = int(os.getenv("MAX_INSTANCES", "100"))

_MODEL: dict | None = None  # cached across warm invocations
_COLD_START = True


class BadRequest(Exception):
    pass


# --------------------------------------------------------------------------- model
def _load_model() -> dict:
    bucket = os.getenv("MODEL_S3_BUCKET", "")
    key = os.getenv("MODEL_S3_KEY", "")
    if bucket and key:
        import boto3  # available in the Lambda runtime; imported lazily for local use

        s3 = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL") or None)
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        model = json.loads(body)
        model["_source"] = f"s3://{bucket}/{key}"
    else:
        model = json.loads(BUNDLED_MODEL_PATH.read_text(encoding="utf-8"))
        model["_source"] = "bundled"
    logger.info(json.dumps({"event": "model_loaded", "version": model["model_version"],
                            "source": model["_source"]}))
    return model


def get_model() -> dict:
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def _softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def predict_one(model: dict, record: dict[str, Any]) -> dict:
    features = model["features"]
    missing = [f for f in features if f not in record]
    if missing:
        raise BadRequest(f"missing feature(s): {', '.join(missing)}")
    try:
        x = [float(record[f]) for f in features]
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"all features must be numeric: {exc}") from exc
    if not all(math.isfinite(v) for v in x):
        raise BadRequest("features must be finite numbers")

    z = [(v - m) / s for v, m, s in zip(x, model["scaler_mean"], model["scaler_scale"])]
    scores = [sum(w * zi for w, zi in zip(row, z)) + b
              for row, b in zip(model["coef"], model["intercept"])]
    probs = _softmax(scores)
    best = max(range(len(probs)), key=probs.__getitem__)
    return {
        "label": model["classes"][best],
        "class_index": best,
        "probabilities": {c: round(p, 4) for c, p in zip(model["classes"], probs)},
    }


# --------------------------------------------------------------------------- events
def _parse_request(event: dict) -> tuple[str, str, Any]:
    """Return (method, path, payload) for API Gateway/function-URL or direct events."""
    http = (event.get("requestContext") or {}).get("http")
    if http:  # API Gateway HTTP API v2 / function URL
        method, path = http.get("method", "POST"), event.get("rawPath", "/predict")
        body = event.get("body")
        if body and event.get("isBase64Encoded"):
            import base64
            body = base64.b64decode(body).decode("utf-8")
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            raise BadRequest(f"body is not valid JSON: {exc.msg}") from exc
        return method, path, payload
    if "httpMethod" in event:  # REST API v1 (in case someone wires it that way)
        body = event.get("body")
        return event["httpMethod"], event.get("path", "/predict"), json.loads(body) if body else {}
    return "POST", "/predict", event  # direct invocation


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def lambda_handler(event: dict, context: Any = None) -> dict:
    global _COLD_START
    started = time.perf_counter()
    cold, _COLD_START = _COLD_START, False
    request_id = getattr(context, "aws_request_id", "local")

    try:
        method, path, payload = _parse_request(event or {})
        model = get_model()

        if method == "GET" and path.endswith("/health"):
            return _response(200, {"status": "ok", "model_version": model["model_version"],
                                   "model_source": model["_source"]})

        if not isinstance(payload, dict):
            raise BadRequest("request body must be a JSON object")
        if "instances" in payload:
            instances = payload["instances"]
        elif "features" in payload:
            instances = [payload["features"]]
        else:
            raise BadRequest('expected "instances": [...] or "features": {...}')
        if not isinstance(instances, list) or not instances:
            raise BadRequest('"instances" must be a non-empty list')
        if len(instances) > MAX_INSTANCES:
            raise BadRequest(f"at most {MAX_INSTANCES} instances per request")

        predictions = [predict_one(model, rec) for rec in instances]
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        logger.info(json.dumps({"event": "prediction", "request_id": request_id, "n": len(predictions),
                                "cold_start": cold, "latency_ms": latency_ms,
                                "model_version": model["model_version"]}))
        return _response(200, {
            "model_version": model["model_version"],
            "predictions": predictions,
            "cold_start": cold,
            "latency_ms": latency_ms,
        })
    except BadRequest as exc:
        logger.warning(json.dumps({"event": "bad_request", "request_id": request_id, "error": str(exc)}))
        return _response(400, {"error": str(exc)})
    except Exception:  # never leak a stack trace to the caller
        logger.exception("unhandled error")
        return _response(500, {"error": "internal error", "request_id": request_id})
