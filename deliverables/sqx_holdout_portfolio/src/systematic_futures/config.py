from __future__ import annotations

from dataclasses import dataclass

MODEL_VERSION = "frozen-2025-08-21-v1"


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str
    strategy_id: str
    bars_valid: int
    exit_after_bars: int
    profit_target_atr_multiple: float
    profit_target_atr_period: int
    stop_loss_atr_multiple: float
    stop_loss_atr_period: int
    tick_size: float
    point_value: float


STRATEGIES: dict[str, StrategyConfig] = {
    "6E": StrategyConfig("6E", "FX-EUR-01", 55, 8, 2.6, 15, 2.1, 15, 0.00005, 125_000.0),
    "6J": StrategyConfig("6J", "FX-JPY-01", 177, 0, 1.2, 30, 3.1, 40, 0.0000005, 12_500_000.0),
    "CL": StrategyConfig("CL", "ENERGY-CL-01", 43, 20, 2.9, 10, 3.5, 50, 0.01, 1_000.0),
    "ES": StrategyConfig("ES", "EQUITY-ES-01", 122, 38, 4.7, 40, 3.1, 30, 0.25, 50.0),
    "GC": StrategyConfig("GC", "METAL-GC-01", 113, 28, 5.5, 40, 1.1, 40, 0.1, 100.0),
    "HG": StrategyConfig("HG", "METAL-HG-01", 198, 40, 2.4, 25, 3.9, 25, 0.0005, 25_000.0),
    "NG": StrategyConfig("NG", "ENERGY-NG-01", 99, 16, 4.0, 20, 3.0, 20, 0.001, 10_000.0),
}
