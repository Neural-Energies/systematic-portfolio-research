"""Point-in-time feature selection, winsorization, and robust scaling."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml


def build_feature_registry(
    metadata: Mapping[str, Any],
    redundancy: pd.DataFrame,
    exact_drops: dict[str, str],
    degenerate_drops: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Create an auditable selection registry without dropping instruments."""
    family_lookup = {
        feature: family
        for family, columns in cast(dict[str, list[str]], metadata["feature_families"]).items()
        for feature in columns
    }
    correlated = set(redundancy["feature_a"]).union(redundancy["feature_b"])
    degenerate_drops = degenerate_drops or {}
    all_drops = {**exact_drops, **degenerate_drops}
    rows = []
    for feature in cast(list[str], metadata["feature_columns"]):
        selected = feature not in all_drops
        if feature in exact_drops:
            status = "drop_exact_duplicate"
            note = "Exact algebraic duplicate; retain the named representative."
        elif feature in degenerate_drops:
            status = "drop_cross_instrument_degenerate"
            note = "Degenerate for one or more instruments; retain the robust proxy."
        else:
            status = "selected"
            note = "Retained pending development-only predictive diagnostics."
        rows.append(
            {
                "feature": feature,
                "family": family_lookup.get(feature, "unclassified"),
                "selected": selected,
                "status": status,
                "duplicate_of": all_drops.get(feature, ""),
                "high_correlation_review": feature in correlated,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def point_in_time_robust_scale(
    features: pd.DataFrame,
    columns: list[str],
    *,
    minimum_history: int = 120,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Winsorize and scale each symbol using only observations before the current row."""
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("winsorization quantiles must be ordered inside [0, 1]")
    result = features.sort_values(["symbol", "trading_date"], ignore_index=True).copy()
    pieces: list[pd.DataFrame] = []
    for _, group in result.groupby("symbol", sort=True, observed=True):
        group = group.copy()
        transformed_columns: dict[str, pd.Series] = {}
        for column in columns:
            values = group[column].astype(float)
            expanding = values.expanding(min_periods=minimum_history)
            lower = expanding.quantile(lower_quantile).shift(1)
            upper = expanding.quantile(upper_quantile).shift(1)
            median = expanding.median().shift(1)
            q25 = expanding.quantile(0.25).shift(1)
            q75 = expanding.quantile(0.75).shift(1)
            standard_deviation = expanding.std(ddof=1).shift(1)
            scale = (q75 - q25).where((q75 - q25).ne(0.0), standard_deviation)
            clipped = values.clip(lower=lower, upper=upper)
            transformed_columns[column] = clipped.sub(median).div(scale.where(scale.ne(0.0)))
        transformed = pd.concat(
            [
                group[["symbol", "trading_date", "feature_available_at_utc"]],
                pd.DataFrame(transformed_columns, index=group.index),
            ],
            axis=1,
        )
        transformed["model_row_complete"] = transformed[columns].notna().all(axis=1)
        pieces.append(transformed)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["trading_date", "symbol"], ignore_index=True
    )


def write_preprocessed_features(
    feature_path: Path = Path("data/processed/features/development_session_features.parquet"),
    metadata_path: Path = Path("data/catalog/development_features.json"),
    redundancy_path: Path = Path("outputs/feature_redundancy_pairs.csv"),
    config_path: Path = Path("config/preprocessing.yaml"),
    output_root: Path = Path("data/processed/model_inputs"),
    catalog_root: Path = Path("data/catalog"),
) -> pd.DataFrame:
    """Freeze selection policy and write a leakage-safe development model matrix."""
    features = pd.read_parquet(feature_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    redundancy = pd.read_csv(redundancy_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["preprocessing"]
    exact_drops = dict(config["redundancy"]["exact_feature_drops"])
    degenerate_drops = dict(config["redundancy"]["cross_instrument_degenerate_drops"])
    registry = build_feature_registry(metadata, redundancy, exact_drops, degenerate_drops)
    selected = registry.loc[registry["selected"], "feature"].tolist()
    processed = point_in_time_robust_scale(
        features,
        selected,
        minimum_history=int(config["minimum_history_sessions"]),
        lower_quantile=float(config["winsorization"]["lower_quantile"]),
        upper_quantile=float(config["winsorization"]["upper_quantile"]),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    catalog_root.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(
        output_root / "development_model_features.parquet", index=False, compression="zstd"
    )
    registry.to_csv(catalog_root / "feature_registry.csv", index=False)
    summary = {
        "input_features": len(metadata["feature_columns"]),
        "selected_features": len(selected),
        "dropped_exact_duplicates": exact_drops,
        "dropped_cross_instrument_degenerate": degenerate_drops,
        "mature_complete_rows": int(processed["model_row_complete"].sum()),
        "preprocessing": config,
        "contains_targets": False,
    }
    (catalog_root / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build point-in-time model features.")
    parser.parse_args()
    processed = write_preprocessed_features()
    print(
        f"Built {len(processed):,} model-feature rows; "
        f"{int(processed['model_row_complete'].sum()):,} are complete."
    )


if __name__ == "__main__":
    main()
