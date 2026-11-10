"""Transform stage: de-duplicate, enforce types and make a reproducible stratified split."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, TARGET, PipelineConfig
from .io_utils import write_csv_atomic

log = logging.getLogger(__name__)


def transform(validated_path: str, run_id: str, cfg: PipelineConfig) -> dict[str, str]:
    df = pd.read_csv(validated_path)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df[FEATURES] = df[FEATURES].astype("float64")
    df[TARGET] = df[TARGET].astype("int64")

    # Fixed random_state => the same input always yields the same split (idempotent).
    train_df, test_df = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.random_state, stratify=df[TARGET]
    )
    out_dir = Path(cfg.run_dir("processed", run_id))
    paths = {
        "train": str(write_csv_atomic(train_df, out_dir / "train.csv")),
        "test": str(write_csv_atomic(test_df, out_dir / "test.csv")),
    }
    log.info("Transform: %d rows (%d duplicates dropped) -> train=%d test=%d",
             len(df), before - len(df), len(train_df), len(test_df))
    return paths
