"""Run the whole pipeline once without Airflow, and refresh the model bundled with the Lambda.

    python scripts/run_pipeline_local.py                 # clean run
    python scripts/run_pipeline_local.py --bad-data      # watch the quality gate stop the run
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import REPO_ROOT, PipelineConfig  # noqa: E402
from pipeline.deploy import deploy  # noqa: E402
from pipeline.extract import extract  # noqa: E402
from pipeline.transform import transform  # noqa: E402
from pipeline.train import train  # noqa: E402
from pipeline.validate import validate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=date.today().isoformat())
    ap.add_argument("--bad-data", action="store_true", help="inject corrupted rows")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = PipelineConfig()
    raw = extract(args.run_id, cfg, inject_bad_rows=args.bad_data)
    validated = validate(raw, args.run_id, cfg)
    split = transform(validated, args.run_id, cfg)
    artefacts = train(split, args.run_id, cfg)
    deploy(artefacts, args.run_id, cfg)

    bundled = REPO_ROOT / "src" / "inference" / "model" / "model.json"
    shutil.copyfile(artefacts["model"], bundled)
    print(f"\nBundled model refreshed: {bundled}")


if __name__ == "__main__":
    main()
