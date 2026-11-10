"""Extract stage: pull the raw dataset and land it, unchanged, in data/raw/<run_id>/."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

from .config import FEATURES, TARGET, PipelineConfig
from .io_utils import write_csv_atomic

log = logging.getLogger(__name__)


def _load_source(source_csv: str | None) -> pd.DataFrame:
    if source_csv:
        log.info("Reading source CSV %s", source_csv)
        return pd.read_csv(source_csv)
    # Default source: UCI Wine dataset bundled with scikit-learn (no network needed).
    bunch = load_wine(as_frame=True)
    df = bunch.frame.rename(columns={"od280/od315_of_diluted_wines": "od280_od315"})
    return df[FEATURES + [TARGET]]


def _corrupt(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Append deliberately bad rows. Used only to demonstrate the data-quality gate failing."""
    rng = np.random.default_rng(seed)
    bad = df.sample(5, random_state=seed).copy()
    bad.loc[bad.index[0], "alcohol"] = np.nan          # missing value
    bad.loc[bad.index[1], "magnesium"] = -40           # physically impossible
    bad.loc[bad.index[2], "hue"] = 99.0                # out of range
    bad.loc[bad.index[3], TARGET] = 7                  # unknown class label
    bad.loc[bad.index[4], "proline"] = rng.uniform(1e5, 1e6)
    return pd.concat([df, bad], ignore_index=True)


def extract(run_id: str, cfg: PipelineConfig, source_csv: str | None = None,
            inject_bad_rows: bool = False) -> str:
    """Idempotent: re-running for the same run_id overwrites the same file with the same content."""
    df = _load_source(source_csv)
    if inject_bad_rows:
        log.warning("inject_bad_rows=True: corrupting data to exercise the quality gate")
        df = _corrupt(df, cfg.random_state)
    out = Path(cfg.run_dir("raw", run_id)) / "wine.csv"
    write_csv_atomic(df, out)
    log.info("Extracted %d rows -> %s", len(df), out)
    return str(out)
