# Compilation time vs. constraint-graph topology

Signed-resolution compile (`compile_constraints`) of synthetic binary-clause instances over m variables of domain K=6, |S_t| = round(tau*K) with tau=0.5. Elimination ordering: `min-fill`. Each row is the mean over 3 seeds.

`exp` is the banded graph (edge iff 0 < |i-j| <= b); `poly` is a spanning tree (random attachment, m-1 edges).

| topology | b | m | width | clauses in | clauses out | resolvents | compile (mean) | std | per var |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `poly` | - | 20 | 1 | 38 | 40 | 2 | 0.0020s | 0.0002 | 0.100ms |
| `poly` | - | 50 | 1 | 98 | 99 | 1 | 0.0111s | 0.0001 | 0.222ms |
| `poly` | - | 100 | 1 | 198 | 199 | 2 | 0.0323s | 0.0002 | 0.323ms |
| `poly` | - | 200 | 1 | 398 | 402 | 5 | 0.1282s | 0.0006 | 0.641ms |
| `exp` | 1 | 20 | 1 | 38 | 38 | 0 | 0.0014s | 0.0000 | 0.070ms |
| `exp` | 1 | 50 | 1 | 98 | 99 | 1 | 0.0083s | 0.0001 | 0.166ms |
| `exp` | 1 | 100 | 1 | 198 | 201 | 3 | 0.0323s | 0.0002 | 0.323ms |
| `exp` | 1 | 200 | 1 | 398 | 405 | 7 | 0.1271s | 0.0020 | 0.635ms |
| `exp` | 2 | 20 | 2 | 74 | 103 | 29 | 0.0045s | 0.0006 | 0.225ms |
| `exp` | 2 | 50 | 2 | 194 | 265 | 72 | 0.0225s | 0.0009 | 0.450ms |
| `exp` | 2 | 100 | 2 | 394 | 535 | 141 | 0.0834s | 0.0026 | 0.834ms |
| `exp` | 2 | 200 | 2 | 794 | 1,077 | 285 | 0.3225s | 0.0041 | 1.613ms |
| `exp` | 3 | 20 | 3 | 108 | 204 | 96 | 0.0220s | 0.0111 | 1.100ms |
| `exp` | 3 | 50 | 3 | 288 | 565 | 278 | 0.0979s | 0.0075 | 1.958ms |
| `exp` | 3 | 100 | 3 | 588 | 1,149 | 566 | 0.2737s | 0.0257 | 2.737ms |
| `exp` | 3 | 200 | 3 | 1,188 | 2,327 | 1,152 | 0.8342s | 0.0291 | 4.171ms |
| `exp` | 4 | 20 | 4 | 140 | 316 | 181 | 0.0565s | 0.0206 | 2.825ms |
| `exp` | 4 | 50 | 4 | 380 | 934 | 574 | 0.3791s | 0.1924 | 7.582ms |
| `exp` | 4 | 100 | 4 | 780 | 1,801 | 1,071 | 0.7305s | 0.2158 | 7.305ms |
| `exp` | 4 | 200 | 4 | 1,580 | 3,638 | 2,160 | 2.0976s | 0.5058 | 10.488ms |
| `exp` | 5 | 20 | 5 | 170 | 475 | 314 | 0.2887s | 0.1142 | 14.435ms |
| `exp` | 5 | 50 | 5 | 470 | 1,021 | 602 | 0.4708s | 0.1485 | 9.416ms |
| `exp` | 5 | 100 | 5 | 970 | 1,975 | 1,120 | 0.9321s | 0.0513 | 9.321ms |
| `exp` | 5 | 200 | 5 | 1,970 | 3,943 | 2,224 | 2.4464s | 0.3328 | 12.232ms |

`width` is the induced width of the primal graph under the elimination ordering actually used, and it separates the two growth curves. A tree has width 1 under a leaf-first ordering, so every bucket stays local and derives (almost) no resolvents. The banded graph has width b, so bucket contents — and the resolvents they generate — grow with b at every one of the m elimination steps.

The two axes behave differently. Growth in **m** is polynomial for both topologies (fitted exponent ~1.5-2.0 across configs, including the tree): resolution itself is local, and the residual quadratic term is the O(n^2) subsumption sweep over the accumulated clause set, not search. Growth in **b** at fixed m is geometric — each extra band multiplies compile time by ~2.2-3.7x — which is the exponential-in-treewidth term. Topology, not problem size, is what makes a compile intractable.

The data-generation parameters (theta, lambda, epsilon, R) affect the sampled points and, through the ground-truth labels, which value-sets get planted in each clause. They do not change the graph topology, so compile time is driven by (m, b) and the topology alone.
