# Target benchmarks and MAE risk sensitivity

12,000 independent one-contract candidate-policy-cost replays. All four matching RTH charts, 100 entry families each, January–August13,2025 development only. Entry/rule/operator choices unchanged. No reserved outcomes or joint target-stop optimization.

Hourly historical forecasts are competitive as exits:63–64of100families profitable versus61withthepreviouslearnedtarget. Learnedhourlyforecasts haveMAE33.99points versus37.76historicalmean, butRMSE70.41 versus70.72—large misses remain. Simple forecasting methods should remain benchmark components.

Hourly learned target multipliers0.5–0.8 retained57–61profitablefamilies;0.9–1.0retained50–51. This supports studying a region of settings instead of claiming one optimal multiplier.

Hourly NQ_000054683951 with originaltarget and MAEstopdistance×0.75:36trades,net$16,655,stress$15,570,PF3.92,win83.3%,minuteDD$2,407.50. Originaltarget/no-stop:net$10,215,DD$4,842.50. Removingbestday leaves$11,165. Q1$3,515/14trades,Q2$14,000/21trades,partialQ3−$860/1trade. This is an exploratory risk candidate, not a promoted configuration.

The same hourly entry with historicalnormalizedMFEq70×0.7target andnostop:net$28,160,stress$27,005,PF2.62,win69.4%,DD$6,085. Existingfrozenlearnedtarget:net$26,360. The target and risk result are separate tests; their profits cannot be combined or added.

Five-minute median results remain negative.15-minute results are mixed. Four-hour results are sensitive to sparse entry counts. Prior selection contamination, related entry families and repeated development tuning remain; no fresh significance or future-profit claim.

## All method summaries

|   duration | policy                        | component   |   positive |   research_flags |   median_profit |   median_delta |   median_drawdown |
|-----------:|:------------------------------|:------------|-----------:|-----------------:|----------------:|---------------:|------------------:|
|          5 | MAE_distance_x0.75            | risk        |          2 |                0 |         -8762.5 |         2570   |          10897.5  |
|          5 | MAE_distance_x1.0             | risk        |          5 |                1 |         -7790   |         3307.5 |          10460    |
|          5 | MAE_distance_x1.25            | risk        |          4 |                1 |         -9330   |         2010   |          11942.5  |
|          5 | MAE_distance_x1.5             | risk        |          6 |                1 |         -8470   |         3020   |          11555    |
|          5 | frozen_target                 | target      |          8 |                0 |         -9430   |         1892.5 |          16470    |
|          5 | historical_mean_target_x0.7   | target      |         10 |                0 |         -9970   |         2537.5 |          15745    |
|          5 | historical_median_target_x0.7 | target      |         10 |                1 |         -9822.5 |         2765   |          13621.2  |
|          5 | historical_q70_target_x0.7    | target      |         10 |                1 |        -10137.5 |         2040   |          15546.2  |
|          5 | learned_target_x0.5           | target      |          4 |                1 |        -11672.5 |         -327.5 |          15266.2  |
|          5 | learned_target_x0.6           | target      |          6 |                1 |        -10585   |         1492.5 |          14042.5  |
|          5 | learned_target_x0.7           | target      |          5 |                1 |        -10612.5 |         1675   |          15052.5  |
|          5 | learned_target_x0.8           | target      |          5 |                1 |        -10480   |         1220   |          15660    |
|          5 | learned_target_x0.9           | target      |          8 |                0 |        -10162.5 |         1195   |          15665    |
|          5 | learned_target_x1.0           | target      |          8 |                0 |         -9430   |         1892.5 |          16470    |
|          5 | original_baseline             | baseline    |          6 |                0 |        -12817.5 |            0   |          16900    |
|         15 | MAE_distance_x0.75            | risk        |         30 |               10 |         -2287.5 |          587.5 |           6221.25 |
|         15 | MAE_distance_x1.0             | risk        |         40 |               15 |         -1492.5 |         1520   |           6222.5  |
|         15 | MAE_distance_x1.25            | risk        |         36 |               24 |         -1437.5 |         1995   |           6290    |
|         15 | MAE_distance_x1.5             | risk        |         37 |               20 |         -2422.5 |         1475   |           6997.5  |
|         15 | frozen_target                 | target      |         34 |               31 |         -6372.5 |        -1520   |          11400    |
|         15 | historical_mean_target_x0.7   | target      |         22 |               17 |         -8767.5 |        -4097.5 |          13512.5  |
|         15 | historical_median_target_x0.7 | target      |         31 |               24 |         -7052.5 |        -2295   |          12362.5  |
|         15 | historical_q70_target_x0.7    | target      |         25 |               18 |         -8497.5 |        -2947.5 |          13087.5  |
|         15 | learned_target_x0.5           | target      |         37 |               33 |         -3660   |         -110   |          11685    |
|         15 | learned_target_x0.6           | target      |         34 |               29 |         -4207.5 |         -587.5 |          11537.5  |
|         15 | learned_target_x0.7           | target      |         34 |               31 |         -6372.5 |        -1520   |          11400    |
|         15 | learned_target_x0.8           | target      |         34 |               30 |         -4850   |         -322.5 |          10788.8  |
|         15 | learned_target_x0.9           | target      |         27 |               24 |         -5632.5 |         -645   |          12850    |
|         15 | learned_target_x1.0           | target      |         29 |               19 |         -7055   |        -1962.5 |          13327.5  |
|         15 | original_baseline             | baseline    |         32 |               20 |         -3675   |            0   |           8712.5  |
|         60 | MAE_distance_x0.75            | risk        |         60 |               47 |          1287.5 |         1857.5 |           3773.75 |
|         60 | MAE_distance_x1.0             | risk        |         54 |               42 |           852.5 |         1225   |           3863.75 |
|         60 | MAE_distance_x1.25            | risk        |         57 |               43 |           740   |         1225   |           4461.25 |
|         60 | MAE_distance_x1.5             | risk        |         56 |               45 |          1100   |         1190   |           4677.5  |
|         60 | frozen_target                 | target      |         61 |               47 |          1605   |         2667.5 |           7128.75 |
|         60 | historical_mean_target_x0.7   | target      |         63 |               47 |          2097.5 |         2592.5 |           6976.25 |
|         60 | historical_median_target_x0.7 | target      |         64 |               49 |          2012.5 |         2092.5 |           6895    |
|         60 | historical_q70_target_x0.7    | target      |         64 |               45 |          2090   |         3925   |           7543.75 |
|         60 | learned_target_x0.5           | target      |         59 |               47 |          1692.5 |         1775   |           6761.25 |
|         60 | learned_target_x0.6           | target      |         61 |               44 |          1147.5 |         2397.5 |           6895    |
|         60 | learned_target_x0.7           | target      |         61 |               47 |          1605   |         2667.5 |           7128.75 |
|         60 | learned_target_x0.8           | target      |         57 |               44 |          1355   |         2917.5 |           7116.25 |
|         60 | learned_target_x0.9           | target      |         51 |               36 |           252.5 |         1837.5 |           8081.25 |
|         60 | learned_target_x1.0           | target      |         50 |               35 |            15   |         1750   |           8467.5  |
|         60 | original_baseline             | baseline    |         43 |               37 |          -322.5 |            0   |           5491.25 |
|        240 | MAE_distance_x0.75            | risk        |         70 |               11 |          2862.5 |            0   |           3350    |
|        240 | MAE_distance_x1.0             | risk        |         64 |               11 |          2510   |            0   |           3510    |
|        240 | MAE_distance_x1.25            | risk        |         67 |               11 |          3417.5 |            0   |           3350    |
|        240 | MAE_distance_x1.5             | risk        |         68 |               10 |          3417.5 |            0   |           3350    |
|        240 | frozen_target                 | target      |         68 |               16 |          3960   |          510   |           6540    |
|        240 | historical_mean_target_x0.7   | target      |         78 |               14 |          4290   |          710   |           6540    |
|        240 | historical_median_target_x0.7 | target      |         71 |               12 |          3210   |            0   |           2790    |
|        240 | historical_q70_target_x0.7    | target      |         78 |               16 |          5575   |         1925   |           6540    |
|        240 | learned_target_x0.5           | target      |         69 |                9 |          2515   |         -645   |           2790    |
|        240 | learned_target_x0.6           | target      |         71 |               11 |          2522.5 |         -430   |           2790    |
|        240 | learned_target_x0.7           | target      |         73 |               13 |          4117.5 |            0   |           2790    |
|        240 | learned_target_x0.8           | target      |         77 |               14 |          4750   |          740   |           6540    |
|        240 | learned_target_x0.9           | target      |         68 |               14 |          3565   |          182.5 |           6540    |
|        240 | learned_target_x1.0           | target      |         68 |               16 |          3960   |          510   |           6540    |
|        240 | original_baseline             | baseline    |         70 |               11 |          3540   |            0   |           3350    |

## Forward forecast errors

|   duration | method            |   forecasts |   MAE_points |   MSE_points |   RMSE_points |   bias_points |   actual_below_forecast |
|-----------:|:------------------|------------:|-------------:|-------------:|--------------:|--------------:|------------------------:|
|          5 | learned           |        5996 |      10.629  |      497.787 |       22.3111 |     -2.88738  |                0.534857 |
|          5 | historical_median |        5996 |      10.6107 |      451.761 |       21.2547 |     -2.69129  |                0.523349 |
|          5 | historical_mean   |        5996 |      11.0897 |      438.041 |       20.9294 |      0.738075 |                0.622415 |
|          5 | historical_q70    |        5996 |      12.608  |      469.835 |       21.6757 |      5.07476  |                0.724817 |
|         15 | learned           |        2018 |      17.6845 |     1523.47  |       39.0316 |     -1.79571  |                0.570862 |
|         15 | historical_median |        2018 |      18.3542 |     1702.34  |       41.2595 |     -5.1525   |                0.536174 |
|         15 | historical_mean   |        2018 |      19.3835 |     1640.84  |       40.5073 |      1.40312  |                0.632805 |
|         15 | historical_q70    |        2018 |      21.3548 |     1688.28  |       41.0886 |      7.29782  |                0.710605 |
|         60 | learned           |         561 |      33.9871 |     4958.09  |       70.4137 |     -3.0557   |                0.597148 |
|         60 | historical_median |         561 |      35.4309 |     5302.24  |       72.8165 |    -10.611    |                0.547237 |
|         60 | historical_mean   |         561 |      37.7616 |     5001.62  |       70.7222 |      3.06907  |                0.657754 |
|         60 | historical_q70    |         561 |      42.6337 |     5158.4   |       71.822  |     15.7692   |                0.762923 |
|        240 | learned           |         176 |      63.8651 |    10129.7   |      100.646  |     -9.90199  |                0.534091 |
|        240 | historical_median |         176 |      62.2827 |     9476.17  |       97.3456 |    -18.8395   |                0.522727 |
|        240 | historical_mean   |         176 |      66.7244 |     9034.32  |       95.049  |      6.23864  |                0.647727 |
|        240 | historical_q70    |         176 |      75.6804 |    10133.8   |      100.667  |     30.8651   |                0.727273 |