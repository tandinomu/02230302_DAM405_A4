import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

EVENTS = Path(__file__).resolve().parent.parent / "events"


def _load(name):
    return json.loads((EVENTS / name).read_text())


def test_direct_invocation_returns_predictions(app):
    resp = app.lambda_handler(_load("direct_invoke.json"), None)
    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert [p["label"] for p in body["predictions"]] == ["cultivar_0", "cultivar_1", "cultivar_2"]
    for p in body["predictions"]:
        assert abs(sum(p["probabilities"].values()) - 1) < 1e-3


def test_api_gateway_event(app):
    resp = app.lambda_handler(_load("apigw_predict.json"), None)
    assert resp["statusCode"] == 200
    assert json.loads(resp["body"])["predictions"][0]["label"] == "cultivar_1"


def test_cold_start_flag_only_on_first_call(app):
    first = json.loads(app.lambda_handler(_load("direct_invoke.json"), None)["body"])
    second = json.loads(app.lambda_handler(_load("direct_invoke.json"), None)["body"])
    assert first["cold_start"] is True and second["cold_start"] is False


def test_health_endpoint(app):
    event = {"rawPath": "/health", "requestContext": {"http": {"method": "GET"}}}
    body = json.loads(app.lambda_handler(event, None)["body"])
    assert body["status"] == "ok" and body["model_source"] == "bundled"


@pytest.mark.parametrize("event, fragment", [
    ({"instances": [{"alcohol": 13.1}]}, "missing feature"),
    ({"something": 1}, "expected"),
    ({"instances": []}, "non-empty"),
    ({"rawPath": "/predict", "requestContext": {"http": {"method": "POST"}}, "body": "{not json"}, "valid JSON"),
])
def test_bad_requests_return_400(app, event, fragment):
    resp = app.lambda_handler(event, None)
    assert resp["statusCode"] == 400
    assert fragment in json.loads(resp["body"])["error"]


def test_non_numeric_feature_is_rejected(app):
    rec = dict(_load("direct_invoke.json")["instances"][0], alcohol="abc")
    assert app.lambda_handler({"features": rec}, None)["statusCode"] == 400


@mock_aws
def test_model_loaded_from_s3_when_configured(app, monkeypatch):
    model = json.loads((Path(app.__file__).parent / "model" / "model.json").read_text())
    model["model_version"] = "from-s3"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="models")
    s3.put_object(Bucket="models", Key="m.json", Body=json.dumps(model))
    monkeypatch.setenv("MODEL_S3_BUCKET", "models")
    monkeypatch.setenv("MODEL_S3_KEY", "m.json")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    body = json.loads(app.lambda_handler(_load("direct_invoke.json"), None)["body"])
    assert body["model_version"] == "from-s3"
