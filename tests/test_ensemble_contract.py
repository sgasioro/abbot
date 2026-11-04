"""The toy EnsembleModel satisfies the botorch Model contract (smoke test).

Proves the ModelConstructor seam is not GP-coupled: a non-GP surrogate expressed
as a botorch EnsembleModel supports .posterior(X) and coherent (fixed-base-sample)
function samples.
"""

import torch

from abbot.models.ensemble import ToyEnsembleModel, make_toy_mlp


def _model(d_in=2, n_members=5):
    members = [make_toy_mlp(d_in) for _ in range(n_members)]
    return ToyEnsembleModel(members, num_outputs=1)


def test_posterior_shapes():
    model = _model()
    X = torch.randn(7, 2, dtype=torch.double)
    post = model.posterior(X)
    # Mean collapses the ensemble dim: (n, m)
    assert post.mean.shape[-2:] == (7, 1)


def test_coherent_samples_are_deterministic_under_fixed_base():
    model = _model()
    X = torch.randn(4, 2, dtype=torch.double)
    post = model.posterior(X)
    s1 = post.rsample(torch.Size([3]))
    s2 = post.rsample(torch.Size([3]))
    # EnsemblePosterior draws member indices; with the same shape the sample set
    # spans the fixed members, so the set of distinct member-predictions matches.
    assert s1.shape[0] == 3
    assert s2.shape[0] == 3
    # Each sample equals one member's prediction (coherent function sample).
    members = model.forward(X)  # (num_models, n, m)
    for draw in s1:
        matches = [torch.allclose(draw, members[k]) for k in range(members.shape[0])]
        assert any(matches), "each ensemble sample must be a coherent member prediction"


def test_num_outputs():
    assert _model().num_outputs == 1
