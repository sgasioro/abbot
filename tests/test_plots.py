"""abbot.plots: the model, progress and check figures for each campaign shape."""

import math

import pytest

pytest.importorskip("matplotlib")

from abbot import BAXPolicy, Campaign, ProblemSpec, plots
from abbot.goal import BandIntersection

pytestmark = pytest.mark.slow  # GP fits


def _texts(fig) -> str:
    out = [fig._suptitle.get_text() if fig._suptitle else ""]
    for ax in fig.axes:
        out += [ax.get_xlabel(), ax.get_ylabel(), ax.get_title()]
        leg = ax.get_legend()
        if leg is not None:
            out += [t.get_text() for t in leg.get_texts()]
    return " | ".join(out)


def _run(c, f, n):
    for _ in range(n):
        (cand,) = c.suggest()
        c.ingest(f(cand), id=cand.id)
    c.suggest()


@pytest.fixture(scope="module")
def band():
    spec = ProblemSpec.model_validate({
        "domain": [{"kind": "continuous", "name": "x", "lower": 0, "upper": 4}],
        "outputs": [{"name": "y", "direction": "OBSERVE"}],
    })
    c = Campaign(spec, BAXPolicy(BandIntersection({"y": (0.4, 0.7)})))
    _run(c, lambda d: {"y": math.sin(d["x"])}, 6)
    return c


@pytest.fixture(scope="module")
def anneal():
    spec = ProblemSpec.model_validate({
        "domain": [
            {"kind": "continuous", "name": "temp", "lower": 150, "upper": 400},
            {"kind": "continuous", "name": "time", "lower": 10, "upper": 3600, "log": True},
            {"kind": "enumerated", "name": "substrate", "values": ["Si", "sapphire"]}],
        "outputs": [{"name": "cond", "direction": "MAXIMIZE"},
                    {"name": "rough", "direction": "OBSERVE", "max_val": 3}],
    })

    def film(d):
        lt = math.log10(d["time"])
        g = math.exp(-((d["temp"] - 330) / 45) ** 2 - ((lt - 2.8) / 0.5) ** 2)
        g *= 1.0 if d["substrate"] == "sapphire" else 0.8
        return {"cond": 1000 * g, "rough": 0.8 + 0.01 * d["temp"] / 10 + 0.5 * lt}

    c = Campaign(spec)
    _run(c, film, 10)
    return c


def test_one_variable_shows_the_band_the_region_and_the_next_design(band):
    fig = plots.model(band)
    (ax,) = fig.axes
    text = _texts(fig)
    for label in ("goal band", "model's goal region", "next design", "latest run",
                  "Model after 6 runs"):
        assert label in text
    assert ax.get_xlabel() == "x" and ax.get_ylabel() == "y"
    ys = [r.outputs.item() for r in band.history.records]
    lo, hi = ax.get_ylim()
    assert lo <= min(ys) and hi >= max(ys)  # the band doesn't clip the runs
    assert "Progress after 6 runs" in _texts(plots.progress(band))


def test_several_variables_slice_each_one_with_the_limited_output(anneal):
    fig = plots.model(anneal)
    assert len(fig.axes) == 2 * 3  # cond and rough (it has a limit) x three variables
    text = _texts(fig)
    assert "slices through" in text and "max rough 3" in text
    assert fig.axes[1].get_xscale() == "log"  # time
    labels = [t.get_text() for t in fig.axes[2].get_xticklabels()]
    assert labels == ["Si", "sapphire"]


def test_two_named_variables_give_mean_and_uncertainty_contours(anneal):
    fig = plots.model(anneal, output="cond", variables=["temp", "time"],
                      at={"substrate": "sapphire"})
    axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    assert len(axes) == 2
    assert "substrate=sapphire" in _texts(fig)


def test_progress_and_check(anneal):
    assert "best so far" in _texts(plots.progress(anneal))
    fig = plots.check(anneal)
    assert "leave-one-out R²" in _texts(fig)


def test_plots_say_what_is_wrong(anneal):
    with pytest.raises(ValueError, match="unknown output"):
        plots.model(anneal, output="yield")
    with pytest.raises(ValueError, match="unknown variable"):
        plots.model(anneal, variables=["pressure"])
    empty = Campaign(anneal.spec)
    with pytest.raises(ValueError, match="No model yet"):
        plots.model(empty)
