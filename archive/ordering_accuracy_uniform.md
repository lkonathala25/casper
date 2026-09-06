# Does assigning high-confidence variables first improve accuracy?

Constraint graph, data and classifier scores are identical across the four orderings within each row group; only the INFER assignment order differs. K=6, tau=0.5, 2 clauses/edge, epsilon=1.0, noise_frac=0.5, noise_dist=uniform, temp=10.0. Accuracy is the fraction of variables assigned their true class.

`base %` is the unconstrained argmax of the classifier scores — the accuracy with no constraint layer at all, identical for every ordering. `Δ %` is each ordering's accuracy relative to it. `seeds` is the number of independently generated instances; all four orderings see the same instance within a seed, which is what makes the contrasts below paired. `min_fill` is CASPER's structural heuristic on the primal graph — the ordering the compiler would pick on its own.

| topology | b |   n |    λ | ordering  | seeds | base % |    Δ % | acc % (mean ± sd) |
|:---------|--:|----:|-----:|:----------|------:|-------:|-------:|------------------:|
| P        | - |  50 | 0.90 | min_fill  |    20 |  78.80 |  -0.10 |      78.70 ± 8.86 |
| P        | - |  50 | 0.90 | high_conf |    20 |  78.80 |  +7.90 |      86.70 ± 5.48 |
| P        | - |  50 | 0.90 | low_conf  |    20 |  78.80 |  -5.00 |      73.80 ± 7.08 |
| P        | - |  50 | 0.90 | random    |    20 |  78.80 |  +0.70 |      79.50 ± 7.89 |
| E        | 2 |  50 | 0.90 | min_fill  |    20 |  78.80 |  +4.80 |      83.60 ± 8.02 |
| E        | 2 |  50 | 0.90 | high_conf |    20 |  78.80 | +13.40 |      92.20 ± 4.85 |
| E        | 2 |  50 | 0.90 | low_conf  |    20 |  78.80 |  -5.60 |      73.20 ± 8.74 |
| E        | 2 |  50 | 0.90 | random    |    20 |  78.80 |  +5.10 |      83.90 ± 8.96 |
| P        | - | 100 | 0.90 | min_fill  |    20 |  79.70 |  +1.00 |      80.70 ± 4.04 |
| P        | - | 100 | 0.90 | high_conf |    20 |  79.70 |  +6.45 |      86.15 ± 2.94 |
| P        | - | 100 | 0.90 | low_conf  |    20 |  79.70 |  -4.60 |      75.10 ± 3.88 |
| P        | - | 100 | 0.90 | random    |    20 |  79.70 |  +1.15 |      80.85 ± 4.91 |
| E        | 2 | 100 | 0.90 | min_fill  |    20 |  79.70 |  +2.00 |      81.70 ± 4.77 |
| E        | 2 | 100 | 0.90 | high_conf |    20 |  79.70 | +12.95 |      92.65 ± 3.22 |
| E        | 2 | 100 | 0.90 | low_conf  |    20 |  79.70 |  -3.55 |      76.15 ± 6.92 |
| E        | 2 | 100 | 0.90 | random    |    20 |  79.70 |  +4.60 |      84.30 ± 5.17 |
| P        | - |  50 | 1.00 | min_fill  |    20 |  57.20 |  +0.20 |      57.40 ± 6.81 |
| P        | - |  50 | 1.00 | high_conf |    20 |  57.20 |  +7.10 |      64.30 ± 3.80 |
| P        | - |  50 | 1.00 | low_conf  |    20 |  57.20 |  -6.20 |      51.00 ± 7.09 |
| P        | - |  50 | 1.00 | random    |    20 |  57.20 |  +1.40 |      58.60 ± 4.50 |
| E        | 2 |  50 | 1.00 | min_fill  |    20 |  57.20 |  +3.90 |      61.10 ± 7.99 |
| E        | 2 |  50 | 1.00 | high_conf |    20 |  57.20 | +18.10 |      75.30 ± 4.60 |
| E        | 2 |  50 | 1.00 | low_conf  |    20 |  57.20 |  -8.10 |      49.10 ± 7.80 |
| E        | 2 |  50 | 1.00 | random    |    20 |  57.20 |  +5.80 |      63.00 ± 7.61 |
| P        | - | 100 | 1.00 | min_fill  |    20 |  58.55 |  -2.40 |      56.15 ± 4.16 |
| P        | - | 100 | 1.00 | high_conf |    20 |  58.55 |  +7.25 |      65.80 ± 2.75 |
| P        | - | 100 | 1.00 | low_conf  |    20 |  58.55 |  -5.10 |      53.45 ± 4.64 |
| P        | - | 100 | 1.00 | random    |    20 |  58.55 |  +0.15 |      58.70 ± 4.21 |
| E        | 2 | 100 | 1.00 | min_fill  |    20 |  58.55 |  +1.70 |      60.25 ± 4.62 |
| E        | 2 | 100 | 1.00 | high_conf |    20 |  58.55 | +18.05 |      76.60 ± 3.30 |
| E        | 2 | 100 | 1.00 | low_conf  |    20 |  58.55 |  -7.60 |      50.95 ± 5.82 |
| E        | 2 | 100 | 1.00 | random    |    20 |  58.55 |  +0.80 |      59.35 ± 4.63 |

## Paired contrasts

Each contrast is computed per instance and then averaged, so the two orderings being compared always saw the identical graph, data and scores. `t` is the paired t-statistic; `w/t/l` counts instances where the first ordering won, tied or lost.

| topology | b |   n |    λ | contrast             | seeds |  w/t/l |      t | Δacc % (mean ± sd) |
|:---------|--:|----:|-----:|:---------------------|------:|-------:|-------:|-------------------:|
| P        | - |  50 | 0.90 | high_conf - low_conf |    20 | 19/1/0 |  +9.79 |      +12.90 ± 5.89 |
| P        | - |  50 | 0.90 | high_conf - random   |    20 | 16/4/0 |  +5.68 |       +7.20 ± 5.67 |
| P        | - |  50 | 0.90 | high_conf - min_fill |    20 | 16/3/1 |  +5.23 |       +8.00 ± 6.84 |
| E        | 2 |  50 | 0.90 | high_conf - low_conf |    20 | 19/1/0 | +10.60 |      +19.00 ± 8.01 |
| E        | 2 |  50 | 0.90 | high_conf - random   |    20 | 14/5/1 |  +4.53 |       +8.30 ± 8.19 |
| E        | 2 |  50 | 0.90 | high_conf - min_fill |    20 | 17/2/1 |  +5.43 |       +8.60 ± 7.08 |
| P        | - | 100 | 0.90 | high_conf - low_conf |    20 | 20/0/0 | +13.44 |      +11.05 ± 3.68 |
| P        | - | 100 | 0.90 | high_conf - random   |    20 | 18/2/0 |  +7.09 |       +5.30 ± 3.34 |
| P        | - | 100 | 0.90 | high_conf - min_fill |    20 | 18/1/1 |  +5.41 |       +5.45 ± 4.50 |
| E        | 2 | 100 | 0.90 | high_conf - low_conf |    20 | 20/0/0 |  +9.95 |      +16.50 ± 7.42 |
| E        | 2 | 100 | 0.90 | high_conf - random   |    20 | 19/1/0 |  +6.14 |       +8.35 ± 6.08 |
| E        | 2 | 100 | 0.90 | high_conf - min_fill |    20 | 19/1/0 |  +9.08 |      +10.95 ± 5.39 |
| P        | - |  50 | 1.00 | high_conf - low_conf |    20 | 19/0/1 |  +7.55 |      +13.30 ± 7.87 |
| P        | - |  50 | 1.00 | high_conf - random   |    20 | 17/2/1 |  +5.28 |       +5.70 ± 4.82 |
| P        | - |  50 | 1.00 | high_conf - min_fill |    20 | 17/2/1 |  +5.08 |       +6.90 ± 6.07 |
| E        | 2 |  50 | 1.00 | high_conf - low_conf |    20 | 20/0/0 | +13.32 |      +26.20 ± 8.80 |
| E        | 2 |  50 | 1.00 | high_conf - random   |    20 | 19/1/0 |  +7.83 |      +12.30 ± 7.03 |
| E        | 2 |  50 | 1.00 | high_conf - min_fill |    20 | 20/0/0 |  +8.40 |      +14.20 ± 7.56 |
| P        | - | 100 | 1.00 | high_conf - low_conf |    20 | 20/0/0 | +11.94 |      +12.35 ± 4.63 |
| P        | - | 100 | 1.00 | high_conf - random   |    20 | 20/0/0 |  +8.36 |       +7.10 ± 3.80 |
| P        | - | 100 | 1.00 | high_conf - min_fill |    20 | 20/0/0 |  +7.83 |       +9.65 ± 5.51 |
| E        | 2 | 100 | 1.00 | high_conf - low_conf |    20 | 20/0/0 | +17.50 |      +25.65 ± 6.56 |
| E        | 2 | 100 | 1.00 | high_conf - random   |    20 | 20/0/0 | +14.01 |      +17.25 ± 5.50 |
| E        | 2 | 100 | 1.00 | high_conf - min_fill |    20 | 20/0/0 | +15.96 |      +16.35 ± 4.58 |

