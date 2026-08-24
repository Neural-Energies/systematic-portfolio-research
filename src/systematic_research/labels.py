"""Separately stored forward labels and development-only walk-forward folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from systematic_research.temporal_validation import load_development_panel


def build_forward_labels(
    panel: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 20)
) -> pd.DataFrame:
    """Create future log returns with explicit realization timestamps."""
    frame = panel.sort_values(["symbol", "trading_date"], ignore_index=True).copy()
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("symbol", sort=True, observed=True):
        group = group.copy()
        close = group["close"].astype(float)
        result = group[["symbol", "trading_date"]].copy()
        for horizon in horizons:
            if horizon < 1:
                raise ValueError("label horizons must be positive")
            result[f"forward_log_return_{horizon}s"] = np.log(close.shift(-horizon).div(close))
            result[f"label_available_at_{horizon}s_utc"] = group["session_close_utc"].shift(
                -horizon
            )
        pieces.append(result)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["trading_date", "symbol"], ignore_index=True
    )


def build_walk_forward_folds(
    trading_dates: pd.Series,
    *,
    minimum_train_sessions: int = 504,
    validation_sessions: int = 63,
    embargo_sessions: int = 5,
) -> pd.DataFrame:
    """Define expanding development folds with a session embargo."""
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates).drop_duplicates().sort_values())
    rows = []
    validation_start_index = minimum_train_sessions + embargo_sessions
    fold = 1
    while validation_start_index < len(dates):
        validation_end_index = min(validation_start_index + validation_sessions - 1, len(dates) - 1)
        train_end_index = validation_start_index - embargo_sessions - 1
        rows.append(
            {
                "fold": fold,
                "train_start": dates[0].date().isoformat(),
                "train_end": dates[train_end_index].date().isoformat(),
                "embargo_start": dates[train_end_index + 1].date().isoformat(),
                "embargo_end": dates[validation_start_index - 1].date().isoformat(),
                "validation_start": dates[validation_start_index].date().isoformat(),
                "validation_end": dates[validation_end_index].date().isoformat(),
                "train_sessions": train_end_index + 1,
                "validation_sessions": validation_end_index - validation_start_index + 1,
            }
        )
        fold += 1
        validation_start_index += validation_sessions
    return pd.DataFrame(rows)


def write_labels_and_folds(
    research_root: Path = Path("data/processed/research"),
    label_root: Path = Path("data/processed/labels"),
    catalog_root: Path = Path("data/catalog"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = load_development_panel(research_root)
    labels = build_forward_labels(panel)
    folds = build_walk_forward_folds(panel["trading_date"])
    label_root.mkdir(parents=True, exist_ok=True)
    catalog_root.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(label_root / "development_forward_labels.parquet", index=False)
    folds.to_csv(catalog_root / "development_walk_forward_folds.csv", index=False)
    return labels, folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Build separate labels and development folds.")
    parser.parse_args()
    labels, folds = write_labels_and_folds()
    print(f"Built {len(labels):,} label rows and {len(folds)} development folds.")


if __name__ == "__main__":
    main()
