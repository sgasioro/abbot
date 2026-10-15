"""The README quickstart runs end-to-end against the frozen public surface.

Keeps the documented first-15-minutes idiom honest: if a Stable-tier name is
renamed or the ask/tell shape changes, this breaks before a new user does.
"""

import pytest
import torch

from abbot import (
    Campaign,
    ContinuousDomain,
    OutputSpec,
    ProblemSpec,
    default_initial_designs,
)

pytestmark = pytest.mark.slow  # GP fits via Campaign suggest/ingest


def test_readme_quickstart_recovers_optimum():
    torch.manual_seed(0)
    spec = ProblemSpec(
        domain=[ContinuousDomain(name="temp", lower=20.0, upper=80.0)],
        outputs=[OutputSpec(name="yield", direction="MAXIMIZE")],
    )
    campaign = Campaign(spec)
    campaign.seed(default_initial_designs(spec))

    def run_in_lab(temp: float) -> float:
        return -((temp - 55.0) ** 2) / 100.0  # smooth, peaks at temp=55

    for _ in range(8):
        x = campaign.suggest(1)[0]
        assert set(x) == {"temp"}
        campaign.ingest({"yield": run_in_lab(x["temp"])})

    best = campaign.result().best
    assert abs(float(best.candidate) - 55.0) < 8.0
