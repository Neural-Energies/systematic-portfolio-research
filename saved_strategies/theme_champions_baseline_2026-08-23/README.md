# Theme Champions Equal Weight — saved strategy

This folder is the permanent home for the fixed six-component portfolio saved on August 23,
2026. Running it does not re-rank the 60-strategy search or choose new champions.

From the project workspace, run:

```powershell
.\.venv\Scripts\python.exe -m systematic_research.saved_strategy_runner
```

Each execution creates a timestamped folder under `runs/` containing component returns, portfolio
returns, calendar-period returns, and `metrics.json`. The metrics include win days, win months, win
quarters, win years, Sharpe, annualized return and volatility, maximum drawdown, and total return.

To use a longer dataset, first build the same development feature and session-panel formats, then
pass their paths with `--features` and `--panel`. The six strategy rules remain fixed.

The default command uses development data only. It does not read the sealed holdout.

## Verified baseline rerun

The authoritative verification run is `runs/20260823T155542Z`. On the saved 226-session
validation windows it reproduced the original result exactly:

- Sharpe: 0.9475847325
- Annualized return: 4.3799456627%
- Annualized volatility: 4.6222205916%
- Maximum drawdown: -1.8623147191%
- Win days: 54.4248% (123/226)
- Win months: 45.4545% (5/11)
- Win quarters: 75.0000% (3/4)

These are development/validation results, not sealed-holdout or live results.
