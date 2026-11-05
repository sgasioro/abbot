"""The UI figure builders are pure (history, spec, model, …) -> dict, so the
dimensionality dispatch is tested here without a browser (the C5 payoff)."""

import pytest
import torch

from abbot.history import History, Record
from abbot.problem import ContinuousDomain, OutputSpec, ProblemSpec
from abbot.ui import figures


def _hist(rows):
    h = History()
    for x, y in rows:
        h.append(
            Record(
                candidate=torch.tensor(x, dtype=torch.double),
                observation=None,
                outputs=torch.tensor(y, dtype=torch.double),
            )
        )
    return h


def test_blank_fig_shape():
    fig = figures.blank_fig("x")
    assert fig["data"] == [] and "layout" in fig


def test_progress_fig_single_objective_is_convergence():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="f", direction="MAXIMIZE")],
    )
    h = _hist([([0.1], [1.0]), ([0.5], [3.0]), ([0.9], [2.0])])
    fig = figures.progress_fig(h, spec)
    assert fig["layout"]["title"]["text"].startswith("Best f")
    assert fig["data"][0]["y"] == [1.0, 3.0, 3.0]  # running max


def test_progress_fig_multiobjective_is_hypervolume():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="a", direction="MAXIMIZE"),
                 OutputSpec(name="b", direction="MAXIMIZE")],
    )
    h = _hist([([0.1], [0.1, 0.9]), ([0.5], [0.5, 0.5]), ([0.9], [0.9, 0.1])])
    fig = figures.progress_fig(h, spec)
    assert fig["layout"]["title"]["text"] == "Dominated hypervolume"


def test_progress_fig_pure_bax_is_target_region_without_model():
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="y", direction="OBSERVE")],
    )
    h = _hist([([0.2], [1.0]), ([0.6], [2.0])])
    # No model yet -> identified 0, collected 2; dispatch must not raise.
    fig = figures.progress_fig(h, spec, policy=None, model=None)
    assert fig["layout"]["title"]["text"] == "Target region discovered"
    assert fig["data"][0]["y"] == [0, 2]


@pytest.mark.slow
def test_model_fig_1d_vs_2d_dispatch():
    from abbot.analysis import canonical_Y
    from abbot.model import ModelListConstructor, SingleTaskGPConstructor

    # 1-D -> posterior band (scatter), 2-D -> contour.
    spec1 = ProblemSpec(
        domain=[ContinuousDomain(name="x", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="f", direction="MAXIMIZE")],
    )
    h1 = _hist([([0.0], [0.0]), ([0.5], [1.0]), ([1.0], [0.0])])
    m1 = SingleTaskGPConstructor(bounds=spec1.bounds_tensor()).build(
        h1.designs.double(), canonical_Y(h1, spec1)
    )
    fig1 = figures.model_fig(h1, spec1, m1, figures.VizSelections())
    assert any(t.get("type") == "scatter" for t in fig1["data"])

    spec2 = ProblemSpec(
        domain=[ContinuousDomain(name="a", lower=0.0, upper=1.0),
                ContinuousDomain(name="b", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="f", direction="MAXIMIZE")],
    )
    h2 = _hist([([0.1, 0.2], [0.3]), ([0.5, 0.5], [1.0]), ([0.9, 0.3], [1.2])])
    m2 = ModelListConstructor(bounds=spec2.bounds_tensor()).build(
        h2.designs.double(), canonical_Y(h2, spec2)
    )
    fig2 = figures.model_fig(h2, spec2, m2, figures.VizSelections())
    assert any(t.get("type") == "contour" for t in fig2["data"])
