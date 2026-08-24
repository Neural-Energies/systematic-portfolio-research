"""Small end-to-end check of the numerical and analytical stack."""

from __future__ import annotations

import duckdb
import numpy as np
import polars as pl


def mean_return() -> float:
    """Calculate a tiny return series through Polars and DuckDB."""
    prices = pl.DataFrame({"price": np.array([100.0, 101.0, 100.5, 102.0])})
    returns = prices.with_columns(pl.col("price").pct_change().alias("return")).drop_nulls()
    connection = duckdb.connect()
    connection.register("sample_returns", returns)
    result = connection.execute("SELECT avg(return) FROM sample_returns").fetchone()
    if result is None:
        raise RuntimeError("Smoke query returned no result")
    return float(result[0])


def main() -> None:
    """Run the environment smoke check."""
    value = mean_return()
    print(f"Research environment OK; sample mean return={value:.8f}")


if __name__ == "__main__":
    main()
