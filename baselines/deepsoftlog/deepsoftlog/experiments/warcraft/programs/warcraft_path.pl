% Warcraft 12x12 s-t reachability via a single symbolic functor.
%
% The recursive transitive-closure encoding (path/2 + undirected/2 + edge/2)
% used by Scallop blows up DSL's top-down SLD search with branching ~4 per
% recursion step (see warcraft_12x12_baseline/dsl_path_smoke.py: N=5 takes
% 56s, N=6 times out). To run at 12x12 we push the path semantics into a
% symbolic functor, the same trick mnist_sum uses with SumFunctorAll.
%
% resnet_node/1     - functor backed by ResNet18; image -> per-node softmax
%                     probabilities, packed flat (total_dim = 580).
% warcraft_path/1   - functor backed by WarcraftPathFunctor; flat node
%                     probabilities -> 2-vector [1-p, p] where p =
%                     P(source ~> sink) under the differentiable adjacency.
%
% With ndim=2 and embedding_metric="dot", `eq(~1, ~warcraft_path(...))`
% scores log(p), so the proof probability of valid(Img) equals p and the
% NLL loss equals -log(p).

eq(X, X).

valid(Img) :- eq(~1, ~warcraft_path(resnet_node(Img))).
