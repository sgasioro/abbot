"""Composite-SBI wiring: turn a PosteriorSampler + ObjectiveReducer into an
``evaluate`` callback for the loop / Campaign.

This is the pre-GP mount point (the plan's status quo): each observation is mapped
to posterior samples over theta, then reduced to a scalar F; the GP sees ``(xi, F)``.
The theta samples are returned alongside F so the loop caches them on the Record's
``aux`` payload (``evaluate -> (outputs, aux)`` seam), letting an annealed/re-scorable
reducer avoid re-sampling.

No ``sbi`` / sbido import here — ``PosteriorSampler`` is structural (sbido's
``AnalyticalGaussianPosterior`` and ``SBIPosteriorSampler`` satisfy it by shape),
so this module stays dependency-free; the heavy wrappers live in interop.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from botorch.acquisition import qLogNoisyExpectedImprovement

from abbot.acq_opt import CandidateOptimizer, Continuous
from abbot.experiment import PosteriorSampler
from abbot.history import History
from abbot.model import SingleTaskGPConstructor
from abbot.problem import ProblemSpec
from abbot.reduce import ObjectiveReducer

logger = logging.getLogger(__name__)


def sbi_evaluate(
    sampler: PosteriorSampler,
    reducer: ObjectiveReducer,
    spec: ProblemSpec,
    *,
    n_samples: int = 2048,
    cache_theta: bool = True,
) -> Callable[[object], object]:
    """Build an ``evaluate(observation) -> (F, theta)`` callback.

    ``F`` is the reduced scalar objective (shape (1,)); ``theta`` is the cached
    posterior sample set (or None if ``cache_theta`` is False). Pass the result as
    ``run_campaign(evaluate=...)`` or ``Campaign.step(evaluate=...)``.
    """

    def evaluate(observation: object) -> object:
        theta = sampler.sample(observation, n_samples)
        f = reducer(theta, spec).reshape(1)
        return (f, theta) if cache_theta else f

    return evaluate


def rescore(history, reducer: ObjectiveReducer, spec: ProblemSpec, **reducer_kwargs):
    """Re-reduce cached theta on every record (e.g. with a new annealing ``beta``).

    Returns a list of scalar F values. Uses each Record's cached ``aux`` theta, so
    no re-sampling — the whole point of caching. Records without cached theta are
    skipped (None in the output).
    """
    out = []
    for r in history.records:
        if r.aux is None:
            out.append(None)
            continue
        theta = r.aux if isinstance(r.aux, torch.Tensor) else torch.as_tensor(r.aux)
        out.append(float(reducer(theta, spec, **reducer_kwargs).reshape(-1)[0]))
    return out


# --- re-scored composite BO (sbido's SBIdO designer) ---------------------------


class RescoredBO:
    """Composite-SBI BO whose objective is re-scored every step from cached theta.

    Each clean record's cached posterior samples (``aux``, see
    :func:`sbi_evaluate`) are reduced afresh with ``reducer(theta, spec,
    **schedule(t))``, a GP is fitted to ``(design, F)`` and noisy log-EI proposes
    the next design. ``t`` counts policy steps after the first ``n_initial``
    (seed) shots, so a schedule — an annealed indicator sharpness, a switch from
    a distance score to an indicator — spans exactly the steps the policy chose.

    The reducer and schedule are the application's: abbot supplies the loop,
    the caching and the BO. ``F`` is maximized. The campaign's own model (fitted
    on the stored, fixed-schedule outputs) is not used for proposals, but still
    drives the denoised end-of-run recommendation.

    The GP infers its noise level: each design is scored from one noisy
    observation, so ``F`` is genuinely noisy and that noise is not the Monte
    Carlo error of the sample mean.
    """

    uq_requirement = "point"  # fits its own surrogate; the campaign's is unused

    def __init__(
        self,
        reducer: ObjectiveReducer,
        schedule: Callable[[int], dict] | None = None,
        *,
        n_initial: int = 0,
        optimizer: CandidateOptimizer | None = None,
        kernel: str = "default",
    ) -> None:
        self.reducer = reducer
        self.schedule = schedule
        self.n_initial = n_initial
        self.optimizer = optimizer or Continuous()
        self.kernel = kernel
        self.last_model = None  # the GP behind the latest proposal (for plotting)

    def targets(self, history: History, spec: ProblemSpec) -> tuple[torch.Tensor, torch.Tensor]:
        """``(X, F)`` for the clean records at the current schedule step."""
        clean = history.clean.records
        if not clean:
            raise RuntimeError(
                f"All {len(history)} shots are flagged as faulted; nothing to fit the "
                "surrogate on. Check the detector threshold or the apparatus."
            )
        if any(r.aux is None for r in clean):
            raise ValueError(
                "RescoredBO needs cached posterior samples on every clean record "
                "(evaluate with sbi_evaluate(..., cache_theta=True))."
            )
        t = len(history) - self.n_initial  # 0 on the first policy step; faults count
        kwargs = self.schedule(t) if self.schedule is not None else {}
        X = torch.stack([r.candidate for r in clean]).double()
        F = torch.stack([
            torch.as_tensor(self.reducer(torch.as_tensor(r.aux), spec, **kwargs)).reshape(())
            for r in clean
        ]).double().unsqueeze(-1)
        return X, F

    def suggest(
        self,
        history: History,
        spec: ProblemSpec,
        model=None,
        n: int = 1,
        *,
        pending: torch.Tensor | None = None,
    ) -> torch.Tensor:
        X, F = self.targets(history, spec)
        gp = SingleTaskGPConstructor(bounds=spec.bounds_tensor(), kernel=self.kernel).build(X, F)
        self.last_model = gp
        X_pending = None if pending is None or not pending.numel() else pending.double()
        acq = qLogNoisyExpectedImprovement(gp, X_baseline=X, X_pending=X_pending)
        return self.optimizer.optimize(acq, spec, n)

    def ingest(self, results) -> None:  # stateless: history carries the state
        return None


# --- trained ``sbi`` posteriors (``abbot[sbi]``) -------------------------------


def _shim_arviz() -> None:
    """sbi 0.22 imports arviz via its MCMC posterior; only DirectPosterior is
    used here. A minimal stand-in keeps arviz's broken transitive imports
    (scipy.signal.gaussian, matplotlib.style.core) from ever loading.
    """
    if "arviz" in sys.modules:
        return
    az = types.ModuleType("arviz")
    az_data = types.ModuleType("arviz.data")
    az_data.InferenceData = type("InferenceData", (), {})
    az.data = az_data
    sys.modules["arviz"] = az
    sys.modules["arviz.data"] = az_data


class TrainedPosterior:
    """A trained ``sbi`` posterior as a :class:`PosteriorSampler`.

    ``transform`` maps an observation to the network's input ``x`` (1, d_x) —
    the application's preprocessing. Samples come back in the network's
    parameter space and are mapped to physical units by ``theta_min``/``theta_max``
    when the network was trained on min-max-scaled theta.
    """

    def __init__(
        self,
        posterior: Any,
        transform: Callable[[Any], torch.Tensor] | None = None,
        theta_min: torch.Tensor | None = None,
        theta_max: torch.Tensor | None = None,
    ) -> None:
        self.posterior = posterior
        self.transform = transform
        self.theta_min = theta_min
        self.theta_max = theta_max

    @classmethod
    def from_dir(
        cls, model_dir: str | Path, transform: Callable[[Any], torch.Tensor] | None = None
    ) -> TrainedPosterior:
        """Load ``posterior.pkl`` (a pickled sbi posterior) and, if present,
        ``y_scaler.json`` (``{"y_min": [...], "y_max": [...]}``, the theta
        scaling) from ``model_dir``.
        """
        model_dir = Path(model_dir)
        _shim_arviz()
        with open(model_dir / "posterior.pkl", "rb") as f:
            posterior = pickle.load(f)  # a trusted local model file
        lo = hi = None
        scaler = model_dir / "y_scaler.json"
        if scaler.exists():
            data = json.loads(scaler.read_text())
            lo = torch.tensor(data["y_min"], dtype=torch.float32)
            hi = torch.tensor(data["y_max"], dtype=torch.float32)
        return cls(posterior, transform, lo, hi)

    def sample(self, observation: Any, n: int = 2048) -> torch.Tensor:
        x = self.transform(observation) if self.transform is not None else observation
        x = torch.as_tensor(x).float().reshape(1, -1)
        samples = self.posterior.sample((n,), x=x).cpu().float()
        if self.theta_min is not None:
            samples = samples * (self.theta_max - self.theta_min) + self.theta_min
        return samples
