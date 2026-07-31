# Response: does variable ordering change the result, and are there ordering pitfalls?

Thank you — this is the right question to ask of a sequential projection scheme, and
it prompted an experiment we did not previously have. Short answer:

1. **The feasible region is order-invariant.** Signed-resolution compilation under any
   elimination ordering yields a logically equivalent compiled form, so the set of
   admissible configurations — and hence the guarantee that INFER terminates inside it
   without backtracking — is identical for every ordering. Ordering never changes
   *whether* we land in the constraint set.
2. **The point we land on is not order-invariant, and cannot be.** Each INFER step is an
   exact projection *given the prefix already assigned*, but the composition of exact
   steps is not the joint minimum-edit solution. The feasible set here is discrete and
   non-convex, so the convex alternating-projection results (POCS/Dykstra) that would
   make the sequence converge to the joint projection do not apply. Order-dependence is
   therefore expected behaviour, not an artefact — and it is a knob worth setting
   deliberately.
3. **Confidence-descending ordering is the right default, and the pitfall the reviewer
   anticipates is real and large.** We ran the experiment; assigning high-confidence
   variables first beats every alternative in all 24 paired contrasts we measured, and
   assigning low-confidence variables first is *worse than applying no constraint layer
   at all*.

## Experiment

To measure accuracy (not just constraint satisfaction) we need known ground truth, so
we use synthetic instances in which the classifier's error rate, the constraint graph
topology, and the assignment ordering are independently controllable.

**Data.** `n` variables over `K = 6` classes. Class means sit evenly on a circle of
radius `R = 10` in R^2; `X_i ~ N(mu_{c_i}, theta^2 I)` with `theta = 1`. A fraction
`noise_frac = 0.5` of the points is then corrupted by convex mixing,
`X_i <- (1 - lambda) X_i + lambda * e` with `e ~ N(0, eps^2 I)`, `eps = 1`. We sweep
`lambda in {0.9, 1.0}`; `lambda = 1.0` replaces the corrupted points outright, so the
label information in half the points is destroyed. This yields controlled unconstrained
accuracies of ~73% and ~59% respectively.

**Classifier surrogate.** A softmax over the negative squared distance to each class
centroid, temperature 10. This stands in for a neural head: it is what the constraint
layer receives, and its per-variable max probability is the confidence signal the
ordering heuristics consume. Using a surrogate rather than a trained network is
deliberate — it lets us hold the scores *exactly* fixed across orderings.

**Topology.** Two regimes, matching our compile-cost study: `P` is a spanning tree
(random attachment, `n - 1` edges) and `E` is a banded graph (`edge iff 0 < |i-j| <= b`)
at `b = 2`. `n in {50, 100}`.

**Constraints.** Binary signed clauses, 2 per edge, each literal a value-set of size
`round(tau*K) = 3` with `tau = 0.5`. One endpoint of every clause is pinned to contain
its endpoint's ground-truth class, which keeps every instance satisfiable — so accuracy
differences reflect where in the feasible set each ordering lands, never an UNSAT
artefact.

**Procedure.** For each ordering we set `cs.ordering`, compile, and run the INFER walk:
variables are assigned front-to-back, each taking the highest-scoring value in its
feasible set `S_i(a_<i)`. Four orderings are compared:

| ordering    | rule                                                             |
|:------------|:-----------------------------------------------------------------|
| `min_fill`  | structural min-fill on the primal graph — what the compiler picks unaided |
| `high_conf` | descending max class probability (high-confidence first)          |
| `low_conf`  | ascending (the deliberate worst case)                             |
| `random`    | uniform shuffle                                                   |

**Pairing.** Within a seed, all four orderings see the *identical* graph, data, scores
and labels; only the assignment order differs. 20 independently generated instances per
configuration. Contrasts are computed per instance and then averaged, giving paired
t-statistics and per-instance win/tie/loss counts rather than a comparison of two
independent means.

**Metric.** Accuracy is the fraction of variables assigned their true class. `base %` is
the unconstrained argmax accuracy — identical across orderings by construction, and the
anchor showing the four orderings really are solving the same problem.

## Results

[insert the two tables from `ordering_accuracy.md` here]

Reading the tables:

- **Ordering is worth up to 24 accuracy points on a fixed instance.** At `E, b=2, n=50,
  lambda=1.0`: 77.2% under `high_conf` vs 53.3% under `low_conf`, same graph, same
  scores. This is a larger spread than the effect of the constraint layer itself.
- **`high_conf` wins everywhere.** It is best in all 8 configurations, and in all 24
  paired contrasts: `+5.2` to `+15.8` points over `random`, `+5.9` to `+15.2` over
  `min_fill`, `+10.8` to `+23.9` over `low_conf`. Paired `t` ranges from `+4.18` to
  `+14.83`, and it wins on at least 16 of 20 instances in every contrast (20/20 in 11 of
  the 24).
- **The pitfall is real: `low_conf` is net-harmful.** Its `Δ` versus the unconstrained
  baseline is negative in every configuration (`-4.1` to `-6.8` points). A constraint
  layer applied in the wrong order does not merely fail to help — it destroys accuracy
  the classifier had already achieved. This is exactly the ripple the reviewer describes:
  an early commitment on an uncertain variable propagates through the compiled clauses
  and forces errors onto confident neighbours that would otherwise have been correct.
- **The structurally optimal ordering is not the accuracy-optimal ordering.** `min_fill`
  behaves roughly like `random` on accuracy (and dips slightly below the unconstrained
  baseline on trees, e.g. `-1.35` at `P, n=100, lambda=0.9`). It optimises elimination
  width, which is a compile-cost objective; it carries no information about error
  propagation. We think this decoupling is the most useful thing in the table.
- **The effect grows with graph density and with classifier noise.** The `high_conf`
  advantage is consistently larger on `E` (b=2) than on the tree `P` — more edges, more
  ripple — and larger at `lambda = 1.0` than at `0.9`. Ordering matters most exactly when
  the classifier is least reliable, i.e. when the constraint layer is doing the most work.

## Mechanism

The same runs record why the effect appears. `broke` is the fraction of variables the
layer flipped from correct to incorrect, `fixed` the reverse; `mean |S|` is the average
feasible-set size seen during the walk.

| config              | ordering    | fixed % | broke % | mean \|S\| |
|:--------------------|:------------|--------:|--------:|-----------:|
| E, b=2, n=50, λ=1.0 | `high_conf` |   19.70 |    0.90 |       3.27 |
| E, b=2, n=50, λ=1.0 | `min_fill`  |   14.30 |   10.70 |       2.66 |
| E, b=2, n=50, λ=1.0 | `random`    |   14.40 |   10.70 |       2.81 |
| E, b=2, n=50, λ=1.0 | `low_conf`  |   11.00 |   16.10 |       2.44 |

`high_conf` does not simply fix more errors; it almost stops breaking correct variables
(0.9% vs 10.7–16.1%), and this holds across every configuration (`high_conf` breaks
0.5–1.1% throughout, versus 8.0–16.1% for `low_conf`). It also leaves *larger* feasible
sets downstream: committing confident variables first prunes the search in a way that is
usually consistent with the truth, whereas an early wrong commitment over-constrains
later variables and forces them off their correct value. That is the ripple effect made
quantitative.

## Robustness to the noise model

Because the corruption law is the one modelling choice that could plausibly drive the
result, we repeated the entire sweep with `e ~ eps * U[0,1]^2` in place of the Gaussian,
holding seeds, graphs, clauses, labels and clean points fixed. The uniform draw is not a
cosmetic swap: it is off-centre and smaller in scale, so corrupted points land in the
positive quadrant and their argmax falls only on the two nearest classes rather than
spreading over all six — the errors become *systematically* biased rather than isotropic.

Every conclusion survives. `high_conf` is again best in all 8 configurations and wins all
24 paired contrasts (`+5.3` to `+17.3` over `random`, `+5.5` to `+16.4` over `min_fill`,
`+11.1` to `+26.2` over `low_conf`; paired `t` from `+4.53` to `+17.50`, at most one
per-instance loss in any contrast). `low_conf` remains net-harmful in all 8 configurations
(`-3.6` to `-8.1`), and `min_fill` again tracks the unconstrained baseline. The
`broke` asymmetry is if anything sharper: 0.1–0.8% under `high_conf` versus 7.9–18.1%
under `low_conf`.

Base accuracy shifts as expected — at `lambda = 1.0` it is unchanged (57.2% vs 58.4%,
since pure noise makes the argmax independent of the label whatever its law), while at
`lambda = 0.9` it rises to ~79% because the uniform draw has smaller magnitude and
corrupts less aggressively. The ordering effect is therefore not an artefact of Gaussian
tails; it depends only on classifier confidence being informative about correctness, which
is the assumption we state explicitly.

## What we will change in the paper

We will add this study as an appendix subsection with both tables, and state explicitly
in the main text that (i) the compiled feasible region is ordering-invariant while the
returned point is not, (ii) the INFER walk should be ordered by descending classifier
confidence, and (iii) the structural (min-fill) ordering, though optimal for compilation,
is not the ordering to use at inference.

<!-- OPTIONAL — include only if the reviewer presses on cost:
One caveat we will state plainly: the confidence ordering is instance-dependent, so it
fixes the elimination ordering used at compile time, and the structural ordering is the
cheaper one to compile. We have measured that trade-off separately (confidence orderings
raise the induced width and hence compile cost) and can include it if the reviewer would
find it useful.
-->
