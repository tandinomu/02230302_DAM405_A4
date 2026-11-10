import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import PipelineConfig  # noqa: E402


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DEPLOY_MODE", "local")
    return PipelineConfig()


@pytest.fixture
def app(monkeypatch):
    """Fresh import of the Lambda handler module (so the model cache starts empty)."""
    monkeypatch.delenv("MODEL_S3_BUCKET", raising=False)
    monkeypatch.delenv("MODEL_S3_KEY", raising=False)
    spec = importlib.util.spec_from_file_location("inference_app", ROOT / "src" / "inference" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
