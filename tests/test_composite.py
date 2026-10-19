"""Composite objectives without any inference: a cached raw measurement re-scored
on a schedule (abbot.composite)."""

import torch

from abbot import (
    Campaign,
    ContinuousDomain,
    History,
    OutputSpec,
    ProblemSpec,
    Record,
    Rescore,
    composite_evaluate,
)

WAVELENGTHS = torch.linspace(500.0, 560.0, 121, dtype=torch.double)
TARGET_NM = 532.0


def _spec() -> ProblemSpec:
    return ProblemSpec(
        domain=[ContinuousDomain(name="temp", lower=0.0, upper=1.0)],
        outputs=[OutputSpec(name="peak_score", direction="MAXIMIZE")],
    )


def _spectrum(peak_nm: float) -> torch.Tensor:
    return torch.exp(-0.5 * ((WAVELENGTHS - peak_nm) / 2.0) ** 2)


def _peak(spectrum: torch.Tensor) -> torch.Tensor:
    return WAVELENGTHS[spectrum.argmax()]


def window_score(spectrum, spec, tol_nm: float = 10.0, sharp: bool = False):
    """Early: minus the distance of the peak from the target (smooth, gives BO a
    gradient). Late: did the peak land within ``tol_nm`` (the real, hit/miss goal)."""
    off = (_peak(spectrum) - TARGET_NM).abs()
    return (off <= tol_nm).double() if sharp else -off


def _schedule(t: int) -> dict:
    # Distance for the first 2 policy steps, then a hit/miss window that narrows.
    return {"sharp": False} if t < 2 else {"sharp": True, "tol_nm": max(10.0 - 3.0 * t, 1.0)}


def test_rescore_anneals_a_raw_measurement_with_no_inference():
    spec = _spec()
    evaluate = composite_evaluate(lambda obs: obs, window_score, spec)  # payload = spectrum
    rescore = Rescore(window_score, _schedule, n_initial=3)
    c = Campaign(spec, targets=rescore)
    peaks = {0.1: 510.0, 0.5: 530.0, 0.9: 545.0}  # temp -> where the peak lands
    for temp, peak in peaks.items():
        _shoot(c, evaluate, temp, peak)

    # Seed phase (t=0): the distance score, as measured.
    assert c.scored_history().clean.outputs.reshape(-1).tolist() == [-22.0, -2.0, -13.0]

    # Three more shots (t=3, tol 1 nm): only the 532.0 nm shot is still a hit.
    for temp, peak in ((0.6, 532.0), (0.7, 535.0), (0.8, 540.0)):
        _shoot(c, evaluate, temp, peak)
    assert rescore.clock(c.history) == 3
    scored = c.scored_history().clean.outputs.reshape(-1).tolist()
    assert scored == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    # The stored outputs keep the measurement-time distance score.
    assert c.history.records[3].outputs.item() == 0.0
    assert c.history.records[1].outputs.item() == -2.0


def _shoot(campaign, evaluate, temp, peak_nm):
    """Measure at ``temp`` (the peak lands at ``peak_nm``) and record the shot."""
    spectrum = _spectrum(peak_nm)
    outputs, payload = evaluate(spectrum)
    campaign.record({"temp": temp}, outputs, observation=spectrum, aux=payload)


def test_rescore_clock_starts_at_zero_and_never_goes_negative():
    rescore = Rescore(window_score, n_initial=3)
    history = History()
    assert rescore.clock(history) == 0
    for _ in range(5):
        history.append(Record(candidate=torch.zeros(1), observation=None))
    assert rescore.clock(history) == 2
