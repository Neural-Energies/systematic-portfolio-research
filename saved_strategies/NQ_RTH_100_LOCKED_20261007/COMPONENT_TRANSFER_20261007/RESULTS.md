# Frozen component transfer across the NQ shortlist

100 entry families on each matching 5-,15-,60-,240-minute RTH chart. Existing component methods were transferred without candidate-specific tuning. Each candidate is simulated independently, one NQ contract, one position at a time. This is not a portfolio.

Evaluation: January1–August13,2025; expanding models fit only earlier permitted labels with five cash sessions purged. No reserved outcomes evaluated. Entries and exits were previously selected, so these are development diagnostics, not blind confirmation.

|   duration | policy      |   positive |   research_candidates |   median_profit |   median_improvement |
|-----------:|:------------|-----------:|----------------------:|----------------:|---------------------:|
|          5 | MAE_only    |          5 |                     1 |         -7790   |               3307.5 |
|          5 | baseline    |          6 |                     0 |        -12817.5 |                  0   |
|          5 | combined    |         11 |                     1 |         -5830   |               5460   |
|          5 | target_only |          8 |                     0 |         -9430   |               1892.5 |
|         15 | MAE_only    |         40 |                    15 |         -1492.5 |               1520   |
|         15 | baseline    |         32 |                    20 |         -3675   |                  0   |
|         15 | combined    |         36 |                    25 |         -2547.5 |               1730   |
|         15 | target_only |         34 |                    31 |         -6372.5 |              -1520   |
|         60 | MAE_only    |         54 |                    42 |           852.5 |               1225   |
|         60 | baseline    |         43 |                    37 |          -322.5 |                  0   |
|         60 | combined    |         53 |                    39 |           460   |               1837.5 |
|         60 | target_only |         61 |                    47 |          1605   |               2667.5 |
|        240 | MAE_only    |         70 |                    11 |          3540   |                  0   |
|        240 | baseline    |         70 |                    11 |          3540   |                  0   |
|        240 | combined    |         68 |                    16 |          3960   |                510   |
|        240 | target_only |         68 |                    16 |          3960   |                510   |

Research flags require at least10 trades, positive base/stress profit, basePF>=1.05, net win>=53%, no unresolved trades. Drawdown, quarters and best-day concentration are shown rather than vetoes. Related rules and policy configurations overlap; do not add counts as independent edges.

Ten-session block resampling gives a conditional paired profit-change interval,2000 draws. It does not correct the original search or subsequent exit selection. A zero-crossing interval means improvement remains uncertain.

## 5-minute research leaders

| candidate       | policy   |   trades |   net_profit |   stress_profit |   profit_factor |   net_win_rate |   minute_close_dd |   profit_delta_to_baseline |   net_without_best_day |   delta_ci_low |   delta_ci_high |
|:----------------|:---------|---------:|-------------:|----------------:|----------------:|---------------:|------------------:|---------------------------:|-----------------------:|---------------:|----------------:|
| NQ_000811385316 | MAE_only |       32 |         2195 |            1225 |         1.80403 |        0.8125  |              1105 |                       1390 |                    820 |         -590   |         4216.25 |
| NQ_000811385316 | combined |       32 |         1220 |             245 |         1.2047  |        0.59375 |              1765 |                        415 |                   -115 |        -3387.5 |         4356    |

## 15-minute research leaders

| candidate       | policy      |   trades |   net_profit |   stress_profit |   profit_factor |   net_win_rate |   minute_close_dd |   profit_delta_to_baseline |   net_without_best_day |   delta_ci_low |   delta_ci_high |
|:----------------|:------------|---------:|-------------:|----------------:|----------------:|---------------:|------------------:|---------------------------:|-----------------------:|---------------:|----------------:|
| NQ_000120523456 | target_only |      174 |        16855 |           11295 |         1.26475 |       0.683908 |            9820   |                       8770 |                   4595 |      -10407    |         37570.2 |
| NQ_000411795799 | combined    |       94 |        15495 |           11815 |         1.56748 |       0.617021 |            6547.5 |                      13380 |                   2600 |       -6435.25 |         39546.5 |
| NQ_000229389917 | combined    |       98 |        15355 |           11560 |         1.56091 |       0.632653 |            6547.5 |                      13960 |                   2460 |       -5141.38 |         36307   |
| NQ_000120523456 | combined    |      174 |        15055 |            8145 |         1.23971 |       0.649425 |           11125   |                       6970 |                   2795 |      -23370.8  |         45141.7 |
| NQ_001005226524 | target_only |      104 |        14050 |           10705 |         1.37901 |       0.625    |            8817.5 |                       9925 |                   2555 |      -10585    |         42241.1 |
| NQ_000124737848 | baseline    |      107 |        13435 |            9915 |         1.66658 |       0.785047 |            4517.5 |                          0 |                  10460 |           0    |             0   |
| NQ_000153555381 | target_only |      142 |        13085 |            8550 |         1.33655 |       0.633803 |            7065   |                       5995 |                   8055 |       -4430.25 |         20767.1 |
| NQ_000124737848 | target_only |      107 |        12325 |            8905 |         1.3323  |       0.663551 |            6385   |                      -1110 |                  -1070 |      -14072.2  |         16305.5 |
| NQ_000998893253 | target_only |       77 |        11865 |            9410 |         1.45071 |       0.649351 |            5912.5 |                       7480 |                   5135 |       -4780.5  |         28500.8 |
| NQ_000152545351 | target_only |      134 |        11535 |            7250 |         1.30359 |       0.626866 |            7065   |                       6425 |                   6505 |       -4560.62 |         22440   |

## 60-minute research leaders

| candidate       | policy      |   trades |   net_profit |   stress_profit |   profit_factor |   net_win_rate |   minute_close_dd |   profit_delta_to_baseline |   net_without_best_day |   delta_ci_low |   delta_ci_high |
|:----------------|:------------|---------:|-------------:|----------------:|----------------:|---------------:|------------------:|---------------------------:|-----------------------:|---------------:|----------------:|
| NQ_000054683951 | target_only |       36 |        26360 |           25230 |         2.82044 |       0.75     |            6085   |                      16145 |                  15080 |       -1145.38 |         43825.9 |
| NQ_000121005444 | target_only |       38 |        24520 |           23325 |         2.3592  |       0.736842 |            6085   |                      18895 |                  14995 |        2659.5  |         45652.9 |
| NQ_000054683951 | combined    |       36 |        19990 |           18905 |         2.11086 |       0.694444 |            6385   |                       9775 |                   8710 |       -4203.25 |         26935.5 |
| NQ_000412452187 | combined    |       29 |        19855 |           18965 |         3.1044  |       0.655172 |            4505   |                      14835 |                   8575 |       -5060.38 |         43590.4 |
| NQ_000124737848 | target_only |       27 |        19690 |           18840 |         3.61661 |       0.703704 |            4185   |                       8335 |                  12605 |       -2956.12 |         27601   |
| NQ_001009785391 | combined    |       30 |        17850 |           16935 |         2.34615 |       0.666667 |            5210   |                      10140 |                   6570 |       -6050.88 |         33812.9 |
| NQ_000412452187 | target_only |       29 |        16780 |           15855 |         2.34133 |       0.655172 |            5130   |                      11760 |                   5500 |       -4855.88 |         36970.7 |
| NQ_000120523456 | target_only |       48 |        16675 |           15150 |         1.59864 |       0.666667 |           11132.5 |                      17390 |                   8955 |        1714.75 |         42766.1 |
| NQ_001005226524 | target_only |       31 |        16450 |           15460 |         2.2336  |       0.741935 |            6895   |                       8465 |                   9320 |      -11621    |         35952.7 |
| NQ_001059060626 | target_only |       33 |        16425 |           15380 |         2.05934 |       0.69697  |            7900   |                      12020 |                   5585 |       -1815.5  |         32842.9 |

## 240-minute research leaders

| candidate       | policy      |   trades |   net_profit |   stress_profit |   profit_factor |   net_win_rate |   minute_close_dd |   profit_delta_to_baseline |   net_without_best_day |   delta_ci_low |   delta_ci_high |
|:----------------|:------------|---------:|-------------:|----------------:|----------------:|---------------:|------------------:|---------------------------:|-----------------------:|---------------:|----------------:|
| NQ_000412452187 | combined    |       11 |        28470 |           28125 |        16.3892  |       0.909091 |              6540 |                      13435 |                  19025 |         395    |         36188.9 |
| NQ_000412452187 | target_only |       11 |        28470 |           28125 |        16.3892  |       0.909091 |              6540 |                      13435 |                  19025 |         395    |         36188.9 |
| NQ_000054683951 | combined    |       15 |        24750 |           24275 |         4.35593 |       0.933333 |             10885 |                       8635 |                  15305 |       -3160.75 |         28155.6 |
| NQ_000054683951 | target_only |       15 |        24750 |           24275 |         4.35593 |       0.933333 |             10885 |                       8635 |                  15305 |       -3160.75 |         28155.6 |
| NQ_000121005444 | combined    |       15 |        24685 |           24210 |         4.31788 |       0.933333 |             13860 |                       8635 |                  15240 |       -3160.75 |         28155.6 |
| NQ_000121005444 | target_only |       15 |        24685 |           24210 |         4.31788 |       0.933333 |             13860 |                       8635 |                  15240 |       -3160.75 |         28155.6 |
| NQ_001059060626 | target_only |       20 |        22935 |           22295 |         2.82458 |       0.75     |             10885 |                       9780 |                  13490 |       -2500.38 |         29150.9 |
| NQ_001059060626 | combined    |       20 |        22935 |           22295 |         2.82458 |       0.75     |             10885 |                       9780 |                  13490 |       -2500.38 |         29150.9 |
| NQ_000117690425 | combined    |       14 |        19065 |           18615 |         3.05221 |       0.857143 |             13860 |                       9105 |                   9620 |       -2467    |         28305.4 |
| NQ_000117690425 | target_only |       14 |        19065 |           18615 |         3.05221 |       0.857143 |             13860 |                       9105 |                   9620 |       -2467    |         28305.4 |