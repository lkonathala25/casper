import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pysdd")
if not torch.cuda.is_available():
    pytest.skip("pypsdd evaluates circuits on cuda only", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
for sub in ("baselines/spl/warcraft", "baselines/spl/grids", "baselines/spl/grids/pypsdd"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))
if str(ROOT / "tasks" / "warcraft") not in sys.path:
    sys.path.append(str(ROOT / "tasks" / "warcraft"))

from compute_mpe import CircuitMPE  # noqa: E402
from layered_circuit import LayeredCircuit  # noqa: E402
from steps_encoding import build_steps_encoding  # noqa: E402
from steps_sdd import build, canonical_edge_ids, load_moves  # noqa: E402
from steps_spl import StepsSplAdapter  # noqa: E402

B, K = 3, 2


@pytest.fixture(scope="module", params=[2, 3])
def setup(request, tmp_path_factory):
    """Compile the g x g steps SDD and load it into SPL's own circuit class."""
    g = request.param
    out = tmp_path_factory.mktemp(f"spl_steps_{g}")
    summary = build(g, out)
    sm = load_moves(g, out)
    cm = CircuitMPE(str(out / f"steps_{g}.vtree"), str(out / f"steps_{g}.sdd"))
    cm.beta.num_reps = K
    cm.beta.mixing = torch.zeros(B, K, device="cuda").log_softmax(1)
    walks = list(build_steps_encoding(g).models())
    return g, summary, StepsSplAdapter(sm), cm, walks


def walk_bits(ad, walk):
    """Move bits, by variable id, of one walk."""
    bits = torch.zeros(ad.sm.num_vars, device="cuda")
    for i in range(len(walk) - 1):
        bits[ad.var_of_move[(i, walk[i], walk[i + 1])] - 1] = 1
    return bits


def brute_scores(ad, lw, walks):
    """(W, B, K) log score of every walk under the literal weights."""
    neg = torch.stack([w[0] for w in lw], 1)
    pos = torch.stack([w[1] for w in lw], 1)
    return torch.stack([torch.where(walk_bits(ad, w).view(1, -1, 1) > 0, pos, neg).sum(1)
                        for w in walks])


def random_litweights(g, ad, seed=0):
    """Literal weights from random per-edge logits."""
    torch.manual_seed(seed)
    logits = torch.randn(B, 2 * g * (g - 1), K, device="cuda")
    return ad.litweights(torch.nn.functional.logsigmoid(logits).clamp(max=-1e-7))


def test_models_are_the_steps_walks(setup):
    """The SDD's model count equals the steps encoding's (2 at 2x2, 402 at 3x3)."""
    g, summary, _, _, walks = setup
    assert summary["model_count"] == len(walks) == {2: 2, 3: 402}[g]


def test_partition_function_matches_enumeration(setup):
    """SPL's circuit Z equals the log-sum-exp of every walk's score."""
    g, _, ad, cm, walks = setup
    lw = random_litweights(g, ad)
    Z = cm.get_tf_ac(lw, log_space=True)
    assert torch.allclose(Z, brute_scores(ad, lw, walks).logsumexp(0), atol=1e-4)


def test_label_round_trip_and_likelihood(setup):
    """Every simple path's label maps to its padded walk and ff_ll scores that walk."""
    g, _, ad, cm, walks = setup
    lw = random_litweights(g, ad, seed=1)
    ids, t = canonical_edge_ids(g), g * g - 1
    scores = brute_scores(ad, lw, walks)
    checked = 0
    for w_idx, walk in enumerate(walks):
        cells = list(walk[: walk.index(t) + 1])
        if len(set(cells)) != len(cells):
            continue
        label = np.zeros(2 * g * (g - 1))
        for a, b in zip(cells, cells[1:]):
            label[ids[(min(a, b), max(a, b))]] = 1
        bits = ad.edge_labels_to_bits(label[None])
        assert tuple(ad.bits_to_walks(bits)[0]) == walk
        assert np.array_equal(ad.bits_to_edges(bits)[0][0].cpu().numpy(), label)
        ll = cm.beta.ff_ll(lw, bits.expand(B, -1))
        assert torch.allclose(ll, scores[w_idx], atol=1e-4)
        checked += 1
    assert checked == {2: 2, 3: 12}[g]


def test_decoder_returns_a_legal_walk(setup):
    """SPL's MPE decoding yields a model of the circuit, i.e. a legal absorbing walk."""
    g, _, ad, cm, walks = setup
    lw = random_litweights(g, ad, seed=2)
    cm.parameterize_ff(lw)
    mpe = cm.get_mpe_inst(B)
    legal = set(walks)
    for walk in ad.bits_to_walks((mpe > 0).long()):
        assert tuple(walk) in legal


def test_layered_engine_matches_pypsdd(setup):
    """LayeredCircuit reproduces pypsdd's Z, ff_cross_entropy gradient and decoded bits."""
    g, _, ad, cm, walks = setup
    lc = LayeredCircuit(cm.beta, ad.sm.num_vars)
    torch.manual_seed(3)
    logits = (3 * torch.randn(B, 2 * g * (g - 1), K, device="cuda")).requires_grad_()
    mixing = torch.randn(B, K, device="cuda").log_softmax(1)
    t = g * g - 1
    simple = [w for w in walks if len(set(w[: w.index(t) + 1])) == w.index(t) + 1]
    target = torch.stack([walk_bits(ad, simple[b % len(simple)]) for b in range(B)])

    lp = torch.nn.functional.logsigmoid(logits).clamp(max=-1e-7)
    cm.beta.mixing = mixing
    ref = cm.ff_cross_entropy(target, ad.litweights(lp)).sum()
    (g_ref,) = torch.autograd.grad(ref, logits)
    lp = torch.nn.functional.logsigmoid(logits).clamp(max=-1e-7)
    neg, pos = ad.lit_tensors(lp)
    ours = lc.cross_entropy(neg, pos, target, mixing).sum()
    (g_ours,) = torch.autograd.grad(ours, logits)
    assert torch.allclose(ours, ref, atol=1e-3)
    assert torch.allclose(g_ours, g_ref, atol=1e-4)

    with torch.no_grad():
        Z_ref = cm.get_tf_ac(ad.litweights(lp), log_space=True)
        assert torch.allclose(lc.log_partition(neg, pos), Z_ref, atol=1e-5)
        cm.parameterize_ff(ad.litweights(lp))
        bits_ref = (cm.get_mpe_inst(B) > 0).long()
        assert torch.equal(lc.mpe_bits(neg, pos, mixing), bits_ref)
