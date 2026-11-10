"""Data-quality gate built with Great Expectations (GX Core 1.x).

The gate runs between extract and transform. If any expectation fails, the
task raises DataQualityError, Airflow marks it failed and nothing downstream
(transform, train, deploy) runs, so bad data can never reach production.
"""
from __future__ import annotations

import logging
from pathlib import Path

import great_expectations as gx
import pandas as pd
from great_expectations.data_context.types.base import ProgressBarsConfig

from .config import FEATURES, TARGET, PipelineConfig
from .io_utils import write_json_atomic

log = logging.getLogger(__name__)
logging.getLogger("great_expectations").setLevel(logging.WARNING)  # keep task logs readable

# Plausible physical ranges for each measurement (loose bounds around the UCI data).
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "alcohol": (10.0, 16.0),
    "malic_acid": (0.5, 7.0),
    "ash": (1.0, 4.0),
    "alcalinity_of_ash": (8.0, 35.0),
    "magnesium": (60.0, 180.0),
    "total_phenols": (0.5, 4.5),
    "flavanoids": (0.2, 6.0),
    "nonflavanoid_phenols": (0.1, 0.8),
    "proanthocyanins": (0.3, 4.0),
    "color_intensity": (1.0, 14.0),
    "hue": (0.3, 2.0),
    "od280_od315": (1.0, 4.5),
    "proline": (200.0, 2000.0),
}


class DataQualityError(Exception):
    """Raised when the dataset fails one or more expectations."""


def build_suite() -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name="wine_raw_suite")
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(column_list=FEATURES + [TARGET])
    )
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=100, max_value=1_000_000))
    for col in FEATURES + [TARGET]:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))
    for col, (lo, hi) in FEATURE_RANGES.items():
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column=col, min_value=lo, max_value=hi))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column=TARGET, value_set=[0, 1, 2]))
    return suite


def run_expectations(df: pd.DataFrame) -> dict:
    """Validate a DataFrame in an ephemeral GX context and return a compact summary."""
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    source = context.data_sources.add_pandas(name="wine_source")
    asset = source.add_dataframe_asset(name="wine_raw")
    batch_def = asset.add_batch_definition_whole_dataframe("whole_dataframe")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    result = batch.validate(build_suite())

    checks = []
    for r in result.results:
        cfg = r.expectation_config
        checks.append({
            "expectation": cfg.type,
            "column": cfg.kwargs.get("column"),
            "success": bool(r.success),
            "unexpected_count": (r.result or {}).get("unexpected_count"),
        })
    return {
        "success": bool(result.success),
        "evaluated": len(checks),
        "failed": sum(not c["success"] for c in checks),
        "checks": checks,
    }


def validate(raw_path: str, run_id: str, cfg: PipelineConfig) -> str:
    df = pd.read_csv(raw_path)
    summary = run_expectations(df)
    report_path = Path(cfg.run_dir("quality", run_id)) / "validation_report.json"
    write_json_atomic({"run_id": run_id, "source": raw_path, **summary}, report_path)

    log.info("Data-quality gate: %d/%d expectations passed (report: %s)",
             summary["evaluated"] - summary["failed"], summary["evaluated"], report_path)
    if not summary["success"]:
        failed = [f'{c["expectation"]}({c["column"]})' for c in summary["checks"] if not c["success"]]
        raise DataQualityError(f"{summary['failed']} expectation(s) failed: {', '.join(failed)}")
    return raw_path
