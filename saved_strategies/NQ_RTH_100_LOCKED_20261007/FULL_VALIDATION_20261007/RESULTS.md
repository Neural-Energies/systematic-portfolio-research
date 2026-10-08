# Historical forward simulation of the100 frozen NQ entries

**13 rules survived, spanning10 similarity groups;10 are preselected group representatives.** The daily-entry requirement is removed. This simulates historical execution; it has not traded a live or paper account.

## What passed

Each survivor has sufficient trade support, positive PNL/PF≥1.10 across9 execution scenarios in both2025 and2026, positive supported calendar quarters, a positive five-session bootstrap lower mean bound, and better expectancy than same-clock/weekday/year eligible random-entry timing. Holm100 diagnostic also passes for these13. These tests do not correct the original ten-million-rule search or make previously seen history a blind holdout.

| Representative | Combined2025–2026 base profit | Base PF | Net win | Worst period/scenario PF | Worst period/scenario profit |
|---|---:|---:|---:|---:|---:|
| NQ_000502850189 | $88,025 | 1.97 | 83.1% | 1.34 | $24,085 |
| NQ_001088325286 | $90,170 | 2.46 | 84.5% | 1.33 | $10,300 |
| NQ_000036419386 | $160,445 | 4.05 | 87.4% | 1.28 | $16,720 |
| NQ_001087785195 | $177,055 | 3.27 | 85.8% | 1.21 | $16,195 |
| NQ_000754524990 | $143,690 | 3.82 | 86.6% | 1.20 | $10,850 |
| NQ_000127938869 | $70,095 | 1.72 | 81.8% | 1.17 | $11,330 |
| NQ_001066794576 | $52,080 | 1.65 | 81.9% | 1.12 | $4,675 |
| NQ_000748828115 | $73,240 | 1.79 | 82.1% | 1.12 | $7,765 |
| NQ_000573223235 | $155,520 | 3.43 | 87.2% | 1.11 | $7,510 |
| NQ_001060537283 | $142,080 | 2.90 | 85.9% | 1.10 | $7,805 |

## Previously highlighted rule

NQ_000926441771 fails the execution-robustness gate.2025 base PNL+$92,995 becomes−$16,295 at two-minute delay (PF0.86), and−$7,735 under combined stress. Its2026 base performance does not override the earlier fragility.

## Scope and limits

2700 period/scenario replays plus900 combined diagnostics,1600 quarter records,200k randomized timing simulations and2000 shared weekly block bootstraps. All300 original base KPI sets reconciled; six independent slow-engine cases and two prefix checks passed. One of967 scheduled cash sessions has one missing minute; none of the selected stress paths became unresolved.

The alternate NQ dataset is the same export, not independent vendor evidence. Per-contract volume-roll history and exporter timestamp semantics remain unverified. Minute data cannot establish actual bid/ask spread, queue priority, latency or stop-fill behavior. Limit penetration and delays are explicit stress assumptions, not observed fills. Session holidays use the cash-session calendar; extended Globex/ML exits remain separate preliminary research. No hard stop or sizing/margin model is added. Similarity groups use80% signal Jaccard connected components and do not imply independent economic ideas.

The source endsAugust20,2026. All post-2024 history was previously used. Thus these13 are simulation survivors for further research, not institutionally certified systems. The10 group representatives were selected using validation worst-scenario PF before the current final gate, without2026 re-ranking.

[Deflated Sharpe research](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) explains why prior search breadth matters. Full-search DSR/PBO is not claimed here because the complete10-million-trial return histories/Sharpe dispersion are not supplied to this test. All100 rules remain saved; rejected rules and exact reasons are retained.
