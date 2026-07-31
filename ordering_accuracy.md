# Does assigning high-confidence variables first improve accuracy?

Constraint graph, data and classifier scores are identical across the four orderings within each row group; only the INFER assignment order differs. K=6, tau=0.5, 2 clauses/edge, epsilon=1.0, noise_frac=0.5, temp=10.0. Accuracy is the fraction of variables assigned their true class.

`base %` is the unconstrained argmax of the classifier scores — the accuracy with no constraint layer at all, identical for every ordering. `Δ %` is each ordering's accuracy relative to it. `seeds` is the number of independently generated instances; all four orderings see the same instance within a seed, which is what makes the contrasts below paired. `min_fill` is CASPER's structural heuristic on the primal graph — the ordering the compiler would pick on its own.

| topology | b |   n |    λ | ordering  | seeds | base % |    Δ % | acc % (mean ± sd) |
|:---------|--:|----:|-----:|:----------|------:|-------:|-------:|------------------:|
| P        | - |  50 | 0.90 | min_fill  |    20 |  72.60 |  +0.80 |      73.40 ± 7.76 |
| P        | - |  50 | 0.90 | high_conf |    20 |  72.60 |  +6.70 |      79.30 ± 4.69 |
| P        | - |  50 | 0.90 | low_conf  |    20 |  72.60 |  -4.10 |      68.50 ± 8.28 |
| P        | - |  50 | 0.90 | random    |    20 |  72.60 |  +1.50 |      74.10 ± 5.64 |
| E        | 2 |  50 | 0.90 | min_fill  |    20 |  72.60 |  +4.10 |      76.70 ± 7.87 |
| E        | 2 |  50 | 0.90 | high_conf |    20 |  72.60 | +14.40 |      87.00 ± 4.28 |
| E        | 2 |  50 | 0.90 | low_conf  |    20 |  72.60 |  -4.50 |      68.10 ± 9.19 |
| E        | 2 |  50 | 0.90 | random    |    20 |  72.60 |  +5.40 |      78.00 ± 7.31 |
| P        | - | 100 | 0.90 | min_fill  |    20 |  73.25 |  -1.35 |      71.90 ± 6.11 |
| P        | - | 100 | 0.90 | high_conf |    20 |  73.25 |  +7.00 |      80.25 ± 4.31 |
| P        | - | 100 | 0.90 | low_conf  |    20 |  73.25 |  -6.10 |      67.15 ± 6.14 |
| P        | - | 100 | 0.90 | random    |    20 |  73.25 |  -1.10 |      72.15 ± 4.63 |
| E        | 2 | 100 | 0.90 | min_fill  |    20 |  73.25 |  +3.85 |      77.10 ± 6.65 |
| E        | 2 | 100 | 0.90 | high_conf |    20 |  73.25 | +13.45 |      86.70 ± 3.05 |
| E        | 2 | 100 | 0.90 | low_conf  |    20 |  73.25 |  -5.25 |      68.00 ± 6.61 |
| E        | 2 | 100 | 0.90 | random    |    20 |  73.25 |  +1.70 |      74.95 ± 7.61 |
| P        | - |  50 | 1.00 | min_fill  |    20 |  58.40 |  -0.40 |      58.00 ± 5.19 |
| P        | - |  50 | 1.00 | high_conf |    20 |  58.40 |  +8.60 |      67.00 ± 4.92 |
| P        | - |  50 | 1.00 | low_conf  |    20 |  58.40 |  -5.90 |      52.50 ± 6.42 |
| P        | - |  50 | 1.00 | random    |    20 |  58.40 |  +0.90 |      59.30 ± 5.16 |
| E        | 2 |  50 | 1.00 | min_fill  |    20 |  58.40 |  +3.60 |      62.00 ± 6.93 |
| E        | 2 |  50 | 1.00 | high_conf |    20 |  58.40 | +18.80 |      77.20 ± 5.13 |
| E        | 2 |  50 | 1.00 | low_conf  |    20 |  58.40 |  -5.10 |      53.30 ± 9.36 |
| E        | 2 |  50 | 1.00 | random    |    20 |  58.40 |  +3.70 |      62.10 ± 9.98 |
| P        | - | 100 | 1.00 | min_fill  |    20 |  59.35 |  -2.35 |      57.00 ± 5.13 |
| P        | - | 100 | 1.00 | high_conf |    20 |  59.35 |  +7.05 |      66.40 ± 4.19 |
| P        | - | 100 | 1.00 | low_conf  |    20 |  59.35 |  -6.80 |      52.55 ± 5.46 |
| P        | - | 100 | 1.00 | random    |    20 |  59.35 |  -1.35 |      58.00 ± 3.81 |
| E        | 2 | 100 | 1.00 | min_fill  |    20 |  59.35 |  +3.95 |      63.30 ± 5.28 |
| E        | 2 | 100 | 1.00 | high_conf |    20 |  59.35 | +16.90 |      76.25 ± 4.17 |
| E        | 2 | 100 | 1.00 | low_conf  |    20 |  59.35 |  -4.95 |      54.40 ± 5.04 |
| E        | 2 | 100 | 1.00 | random    |    20 |  59.35 |  +1.15 |      60.50 ± 5.89 |

## Paired contrasts

Each contrast is computed per instance and then averaged, so the two orderings being compared always saw the identical graph, data and scores. `t` is the paired t-statistic; `w/t/l` counts instances where the first ordering won, tied or lost.

| topology | b |   n |    λ | contrast             | seeds |  w/t/l |      t | Δacc % (mean ± sd) |
|:---------|--:|----:|-----:|:---------------------|------:|-------:|-------:|-------------------:|
| P        | - |  50 | 0.90 | high_conf - low_conf |    20 | 19/0/1 |  +6.81 |      +10.80 ± 7.09 |
| P        | - |  50 | 0.90 | high_conf - random   |    20 | 17/2/1 |  +5.78 |       +5.20 ± 4.02 |
| P        | - |  50 | 0.90 | high_conf - min_fill |    20 | 16/2/2 |  +4.18 |       +5.90 ± 6.31 |
| E        | 2 |  50 | 0.90 | high_conf - low_conf |    20 | 19/0/1 |  +8.54 |      +18.90 ± 9.89 |
| E        | 2 |  50 | 0.90 | high_conf - random   |    20 | 16/0/4 |  +4.75 |       +9.00 ± 8.47 |
| E        | 2 |  50 | 0.90 | high_conf - min_fill |    20 | 19/0/1 |  +6.04 |      +10.30 ± 7.63 |
| P        | - | 100 | 0.90 | high_conf - low_conf |    20 | 20/0/0 |  +9.48 |      +13.10 ± 6.18 |
| P        | - | 100 | 0.90 | high_conf - random   |    20 | 19/1/0 |  +7.82 |       +8.10 ± 4.63 |
| P        | - | 100 | 0.90 | high_conf - min_fill |    20 | 20/0/0 |  +6.41 |       +8.35 ± 5.82 |
| E        | 2 | 100 | 0.90 | high_conf - low_conf |    20 | 20/0/0 | +14.83 |      +18.70 ± 5.64 |
| E        | 2 | 100 | 0.90 | high_conf - random   |    20 | 19/0/1 |  +7.90 |      +11.75 ± 6.65 |
| E        | 2 | 100 | 0.90 | high_conf - min_fill |    20 | 19/1/0 |  +7.44 |       +9.60 ± 5.77 |
| P        | - |  50 | 1.00 | high_conf - low_conf |    20 | 20/0/0 |  +9.95 |      +14.50 ± 6.52 |
| P        | - |  50 | 1.00 | high_conf - random   |    20 | 18/1/1 |  +6.38 |       +7.70 ± 5.40 |
| P        | - |  50 | 1.00 | high_conf - min_fill |    20 | 18/2/0 |  +7.97 |       +9.00 ± 5.05 |
| E        | 2 |  50 | 1.00 | high_conf - low_conf |    20 | 19/1/0 |  +9.82 |     +23.90 ± 10.89 |
| E        | 2 |  50 | 1.00 | high_conf - random   |    20 | 16/1/3 |  +5.41 |     +15.10 ± 12.47 |
| E        | 2 |  50 | 1.00 | high_conf - min_fill |    20 | 20/0/0 |  +9.39 |      +15.20 ± 7.24 |
| P        | - | 100 | 1.00 | high_conf - low_conf |    20 | 20/0/0 |  +9.26 |      +13.85 ± 6.69 |
| P        | - | 100 | 1.00 | high_conf - random   |    20 | 20/0/0 | +10.18 |       +8.40 ± 3.69 |
| P        | - | 100 | 1.00 | high_conf - min_fill |    20 | 20/0/0 |  +7.09 |       +9.40 ± 5.93 |
| E        | 2 | 100 | 1.00 | high_conf - low_conf |    20 | 20/0/0 | +13.33 |      +21.85 ± 7.33 |
| E        | 2 | 100 | 1.00 | high_conf - random   |    20 | 20/0/0 | +11.88 |      +15.75 ± 5.93 |
| E        | 2 | 100 | 1.00 | high_conf - min_fill |    20 | 20/0/0 | +10.72 |      +12.95 ± 5.40 |

