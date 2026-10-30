"""Model diagnostics: is the surrogate a fair picture of the data?

:func:`check` (and ``Campaign.check()``) looks at a fitted campaign and reports, in
plain words, what's worth a second look:

- **Leave-one-out fit.** How well each run is predicted by the model fitted to the
  others, with the fitted settings held fixed (exact for a Gaussian process, so no
  refits). Gives an R² per output and flags runs far from their prediction: a fault
  that slipped through, an outlier, or a sharp feature the model smooths over.
- **Noise.** How much of each output's variation the model reads as noise.
- **Lengthscales.** How far each input must move, as a fraction of its range, for
  an output to change. A very short one usually means the fit is chasing noise or
  an outlier; a very long one, that the output hasn't shown a dependence on it.
- **Runs vs. inputs.** Too few runs for the number of inputs leaves the model
  mostly guessing.

The checks read any BoTorch exact GP (the builtin constructors, model lists,
categorical and source-aware GPs). For other models only the run counts apply.
The thresholds are rules of thumb, not tests: a finding is a prompt to look.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from abbot.history import History
from abbot.problem import ProblemSpec

if TYPE_CHECKING:
    from botorch.models.model import Model

# A run more than this many standard deviations from its leave-one-out prediction
# is flagged. Calibrated, about 1 run in 370 would be. Not higher: the fit absorbs
# part of an outlier (as extra noise), so a run 100 noise levels off scored 3.5-3.7
# on a 2-input bowl.
OUTLIER_Z = 3.0

# Below this leave-one-out R², the model predicts an output poorly.
WEAK_R2 = 0.5

# Mean squared leave-one-out z above this: the model is more confident than its
# errors justify. (1 when calibrated.)
OVERCONFIDENT_Z2 = 4.0

# Above this fraction of the variation read as noise, the model explains little.
NOISY_FRACTION = 0.5

# Lengthscales, as fractions of an input's range (inputs are modeled on [0, 1]).
# Long means nearly flat across the range, and also much longer than the output's
# shortest: BoTorch's default prior alone puts lengthscales near 4·sqrt(inputs), so
# early on every input looks long.
SHORT_LENGTHSCALE = 0.05
LONG_LENGTHSCALE = 10.0
LONG_RATIO = 10.0

# Leave-one-out needs a few runs to mean anything.
MIN_RUNS_FOR_LOO = 5


@dataclass
class Finding:
    """One thing worth a look. ``level`` is ``"warning"`` (likely hurting the
    campaign) or ``"note"`` (worth knowing)."""

    level: str
    message: str
    output: str | None = None


@dataclass
class OutputFit:
    """How the model fits one output.

    ``z`` holds each clean run's leave-one-out residual in standard deviations
    (``(observed - predicted) / predicted std``; positive means the run came out
    higher than predicted), keyed by its step in the history. ``noise_fraction`` is the share of the output's
    variation the model reads as noise; ``lengthscales`` maps each input to its
    lengthscale as a fraction of its range.
    """

    name: str
    r2: float | None = None
    z: dict[int, float] = field(default_factory=dict)
    noise_fraction: float | None = None
    lengthscales: dict[str, float] = field(default_factory=dict)


@dataclass
class CheckReport:
    """What :func:`check` found. Print it for a summary; ``findings`` lists the
    warnings and notes, ``fits`` the numbers per output."""

    n_runs: int
    n_failed: int
    n_inputs: int
    model: str | None
    fits: list[OutputFit] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warning"]

    @property
    def ok(self) -> bool:
        """No warnings (notes allowed)."""
        return not self.warnings

    def __str__(self) -> str:
        failed = f" ({self.n_failed} failed)" if self.n_failed else ""
        lines = [(
            f"Campaign check: {self.n_runs} runs{failed}, {self.n_inputs} inputs, "
            f"model {self.model or 'none'}"
        )]
        for fit in self.fits:
            parts = []
            if fit.r2 is not None:
                parts.append(f"leave-one-out R² {fit.r2:.2f}")
            if fit.noise_fraction is not None:
                parts.append(f"noise {fit.noise_fraction:.0%} of variation")
            if fit.lengthscales:
                ls = ", ".join(f"{k} {v:.2g}" for k, v in fit.lengthscales.items())
                parts.append(f"lengthscales {ls}")
            if parts:
                lines.append(f"  {fit.name}: " + "; ".join(parts))
        for level, title in (("warning", "Warnings"), ("note", "Notes")):
            found = [f for f in self.findings if f.level == level]
            if found:
                lines.append(title)
                lines += [f"  - {f.message}" for f in found]
        if not self.findings:
            lines.append("No problems found.")
        return "\n".join(lines)

    __repr__ = __str__


def check(history: History, spec: ProblemSpec, model: Model | None) -> CheckReport:
    """Diagnose ``model`` (fitted to ``history``'s clean runs, as
    ``Campaign.current_model()`` is) against the data. See the module docstring."""
    clean = history.clean
    steps = [i for i, r in enumerate(history.records) if not r.fault]
    n, d = len(clean), len(spec.inputs)
    report = CheckReport(
        n_runs=len(history), n_failed=len(history) - n, n_inputs=d,
        model=type(model).__name__ if model is not None else None,
    )
    add = report.findings.append

    if len(history) and report.n_failed / len(history) > 0.2:
        add(Finding("note", f"{report.n_failed} of {len(history)} runs failed or were "
                            "rejected; they're left out of the model."))
    if n < d + 1:
        add(Finding("warning", f"{n} clean runs for {d} inputs: the model is mostly "
                               "guessing until there are more runs than inputs."))
    elif n < 2 * d:
        add(Finding("note", f"{n} clean runs for {d} inputs: expect a rough model "
                            f"until about {2 * d}."))
    if model is None:
        if n:
            add(Finding("warning", "No model: the fit failed (see the log)."))
        return report

    gps = _per_output(model, spec.n_outputs)
    if gps is None:
        add(Finding("note", f"Leave-one-out, noise and lengthscale checks need a "
                            f"Gaussian process; {report.model} isn't one."))
        return report

    signs = spec.output_signs(dtype=torch.double).tolist()
    for name, sign, (gp, row) in zip(spec.output_names, signs, gps):
        fit = OutputFit(name=name)
        report.fits.append(fit)
        try:
            mean, var, y, noise = _loo(gp, row)
        except Exception:  # noqa: BLE001 - a model we can't read: say so, not crash
            add(Finding("note", f"{name}: couldn't read the fitted model for "
                                "leave-one-out.", name))
            continue
        fit.lengthscales = _lengthscales(gp, row, spec)
        # The model is in maximize-canonical space; z is reported in raw terms.
        _judge(fit, sign * mean, var, sign * y, noise,
               steps if len(steps) == len(y) else None, add)
    return report


def _per_output(model, m: int):
    """``[(gp, batch row or None)]`` per output, or None if not exact GPs."""
    from botorch.models import ModelListGP
    from gpytorch.models import ExactGP

    if isinstance(model, ModelListGP):
        if len(model.models) != m or not all(isinstance(g, ExactGP) for g in model.models):
            return None
        return [(g, None) for g in model.models]
    if not isinstance(model, ExactGP):
        return None
    if m == 1:
        return [(model, None)]
    if model.train_targets.dim() == 2 and model.train_targets.shape[0] == m:
        return [(model, j) for j in range(m)]  # a batched multi-output GP
    return None


def _loo(gp, row: int | None):
    """Leave-one-out mean and variance at each training point, the targets and the
    noise variance, in the model's standardized space.

    With the fitted settings held fixed, the exact identities are
    ``mean_i = y_i - [K⁻¹ r]_i / [K⁻¹]_ii`` and ``var_i = 1 / [K⁻¹]_ii`` (``K`` the
    prior covariance plus noise, ``r`` the targets minus the prior mean).
    """
    was_training = gp.training
    with torch.no_grad():
        gp.train()  # the prior at the training inputs (inputs transformed as in fitting)
        try:
            f = gp.forward(*gp.train_inputs)
            prior = gp.likelihood(f)
        finally:
            gp.train(was_training)
        K, mu, y = prior.covariance_matrix, prior.mean, gp.train_targets
        noise = torch.diagonal(K - f.covariance_matrix, dim1=-2, dim2=-1)
        if row is not None:
            K, mu, y, noise = K[row], mu[row], y[row], noise[row]
        Kinv = torch.cholesky_inverse(torch.linalg.cholesky(K))
        diag = torch.diagonal(Kinv)
        var = 1.0 / diag
        mean = y - (Kinv @ (y - mu)) / diag
        prior_var = torch.diagonal(K) - noise
    return mean, var, y, (noise.mean(), prior_var.mean())


def _judge(fit: OutputFit, mean, var, y, noise, steps, add) -> None:
    name = fit.name
    noise_var, signal_var = (float(v) for v in noise)
    if noise_var + signal_var > 0:
        fit.noise_fraction = noise_var / (noise_var + signal_var)
        if fit.noise_fraction > NOISY_FRACTION:
            add(Finding("warning", f"{name}: the model reads {fit.noise_fraction:.0%} of "
                                   "the variation as noise, so it explains little. Either "
                                   "the output is that noisy, or it varies in a way the "
                                   "model can't follow.", name))
    shortest = min(fit.lengthscales.values(), default=None)
    for var_name, ls in fit.lengthscales.items():
        if ls < SHORT_LENGTHSCALE:
            add(Finding("warning", f"{name}: changes over {ls:.1%} of {var_name}'s range; "
                                   "the fit may be chasing noise or an outlier.", name))
        elif ls > LONG_LENGTHSCALE and ls > LONG_RATIO * shortest:
            add(Finding("note", f"{name}: hasn't shown a dependence on {var_name} yet.",
                        name))

    n = len(y)
    if n < MIN_RUNS_FOR_LOO:
        return
    z = (y - mean) / var.sqrt()
    labels = steps if steps is not None else list(range(n))
    fit.z = {int(s): float(v) for s, v in zip(labels, z)}
    spread = float(((y - y.mean()) ** 2).sum())
    if spread > 0:
        fit.r2 = 1.0 - float(((y - mean) ** 2).sum()) / spread
        if fit.r2 < 0:
            add(Finding("warning", f"{name}: leave-one-out R² {fit.r2:.2f}; the model "
                                   "predicts unseen runs no better than their average.",
                        name))
        elif fit.r2 < WEAK_R2:
            add(Finding("note", f"{name}: leave-one-out R² {fit.r2:.2f}; the model "
                                "predicts unseen runs only roughly.", name))
    far = sorted((s for s, v in fit.z.items() if abs(v) > OUTLIER_Z),
                 key=lambda s: -abs(fit.z[s]))
    if far:
        which = ", ".join(f"run {s} ({fit.z[s]:+.1f}σ)" for s in far[:5])
        more = f" and {len(far) - 5} more" if len(far) > 5 else ""
        verb = "lies" if len(far) == 1 else "lie"
        add(Finding("warning", f"{name}: {which}{more} {verb} far from what the other "
                               "runs predict: a fault, an outlier, or a sharp feature.",
                    name))
    mean_z2 = float((z**2).mean())
    if mean_z2 > OVERCONFIDENT_Z2:
        add(Finding("warning", f"{name}: leave-one-out errors are {mean_z2 ** 0.5:.1f}× "
                               "what the model expects, so it's overconfident.", name))


def _lengthscales(gp, row: int | None, spec: ProblemSpec) -> dict[str, float]:
    """Each named input's shortest lengthscale across the GP's RBF and Matérn
    kernels, as a fraction of its (modeled) range. Categorical kernels are left out:
    their lengthscales aren't distances."""
    from gpytorch.kernels import MaternKernel, RBFKernel

    names = spec.variable_names
    categorical = set(spec.categorical_indices())
    out: dict[int, float] = {}
    for module in gp.covar_module.modules():
        if not isinstance(module, (RBFKernel, MaternKernel)):
            continue
        ls = module.lengthscale.detach()
        ls = ls.reshape(-1, ls.shape[-1])
        ls = ls[row] if row is not None and ls.shape[0] > row else ls[0]
        dims = (module.active_dims.tolist() if module.active_dims is not None
                else list(range(ls.shape[-1])))
        if len(dims) != ls.shape[-1]:
            continue
        for j, v in zip(dims, ls.tolist()):
            if j < len(names) and j not in categorical:
                out[j] = min(out.get(j, float("inf")), v)
    return {names[j]: out[j] for j in sorted(out)}
