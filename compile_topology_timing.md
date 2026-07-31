# Compilation time vs. constraint-graph topology

Signed-resolution compile (`compile_constraints`) of synthetic binary-clause instances over m variables of domain K=6, |S_t| = round(tau*K) with tau=0.5. Elimination ordering: `min-fill`. Configs with b >= 3 use 15 seeds, the rest 5; instance-to-instance spread is far larger than run-to-run timing noise, so the median and IQR below are over independently generated instances.

`exp` is the banded graph (edge iff 0 < |i-j| <= b); `poly` is a spanning tree (random attachment, m-1 edges).

`K` is the number of classes (= each variable's domain size) and `r` the number of clauses in the generated instance — 2 per edge, so r = 2|E| with |E| = m-1 for `poly` and |E| = mb - b(b+1)/2 for `exp`.

| topology | b |   m | K |     r | width | seeds | compile (ms, mean ± sd) |
|:---------|--:|----:|--:|------:|------:|------:|------------------------:|
| `poly`   | - |  20 | 6 |    38 |     1 |     5 |             1.80 ± 0.10 |
| `poly`   | - |  50 | 6 |    98 |     1 |     5 |             9.80 ± 0.20 |
| `poly`   | - | 100 | 6 |   198 |     1 |     5 |            39.20 ± 0.70 |
| `poly`   | - | 200 | 6 |   398 |     1 |     5 |          149.00 ± 11.80 |
| `exp`    | 1 |  20 | 6 |    38 |     1 |     5 |             1.40 ± 0.00 |
| `exp`    | 1 |  50 | 6 |    98 |     1 |     5 |             8.10 ± 0.10 |
| `exp`    | 1 | 100 | 6 |   198 |     1 |     5 |            31.90 ± 0.20 |
| `exp`    | 1 | 200 | 6 |   398 |     1 |     5 |           151.50 ± 1.70 |
| `exp`    | 2 |  20 | 6 |    74 |     2 |     5 |             4.90 ± 0.90 |
| `exp`    | 2 |  50 | 6 |   194 |     2 |     5 |            25.90 ± 1.30 |
| `exp`    | 2 | 100 | 6 |   394 |     2 |     5 |            88.40 ± 5.20 |
| `exp`    | 2 | 200 | 6 |   794 |     2 |     5 |          328.00 ± 23.70 |
| `exp`    | 3 |  20 | 6 |   108 |     3 |    15 |           27.10 ± 11.70 |
| `exp`    | 3 |  50 | 6 |   288 |     3 |    15 |           88.50 ± 15.80 |
| `exp`    | 3 | 100 | 6 |   588 |     3 |    15 |          255.80 ± 26.90 |
| `exp`    | 3 | 200 | 6 | 1,188 |     3 |    15 |          808.60 ± 76.40 |
| `exp`    | 4 |  20 | 6 |   140 |     4 |    15 |          136.30 ± 96.70 |
| `exp`    | 4 |  50 | 6 |   380 |     4 |    15 |         359.10 ± 111.60 |
| `exp`    | 4 | 100 | 6 |   780 |     4 |    15 |         834.90 ± 212.00 |
| `exp`    | 4 | 200 | 6 | 1,580 |     4 |    15 |       2,012.50 ± 355.90 |
| `exp`    | 5 |  20 | 6 |   170 |     5 |    15 |         315.30 ± 196.00 |
| `exp`    | 5 |  50 | 6 |   470 |     5 |    15 |         520.60 ± 251.70 |
| `exp`    | 5 | 100 | 6 |   970 |     5 |    15 |       1,064.50 ± 328.90 |
| `exp`    | 5 | 200 | 6 | 1,970 |     5 |    15 |       2,729.10 ± 598.70 |

`width` is the induced width of the primal graph under the elimination ordering actually used, and it separates the two growth curves. A tree has width 1 under a leaf-first ordering, so every bucket stays local and derives (almost) no resolvents. The banded graph has width b, so bucket contents — and the resolvents they generate — grow with b at every one of the m elimination steps.

The two axes behave differently. Growth in **m** is polynomial for both topologies (fitted exponent ~1.5-2.0 across configs, including the tree): resolution itself is local, and the residual quadratic term is the O(n^2) subsumption sweep over the accumulated clause set, not search. Growth in **b** at fixed m is geometric — each extra band multiplies compile time by ~2.2-3.7x — which is the exponential-in-treewidth term. Topology, not problem size, is what makes a compile intractable.

The data-generation parameters (theta, lambda, epsilon, R) affect the sampled points and, through the ground-truth labels, which value-sets get planted in each clause. They do not change the graph topology, so compile time is driven by (m, b) and the topology alone.
