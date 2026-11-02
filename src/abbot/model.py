"""The ModelConstructor seam: ``(X, Y, Yvar) -> botorch.Model``.

This is the key SBI seam. BoTorch's ``Model``/``Posterior`` ABC already *is* the
surrogate-agnostic layer, so abbot does NOT define a bespoke Posterior protocol —
a constructor just builds a botorch ``Model`` and every policy composes over its
``.posterior(X)`` unchanged.

Builtins cover single-output GP, multi-output ``ModelListGP``, and a
multi-fidelity GP for specs with source variables. Involved SBI
construction (correlated ``MultiTaskGP``/LCM, per-shot posterior variance via
``train_Yvar``, custom GPyTorch likelihoods) plugs in here WITHOUT touching the
loop or the policy. NN surrogates are botorch ``EnsembleModel`` subclasses (see
``abbot.models.ensemble``), not a parallel interface.

Modelled on Xopt's ``ModelConstructor`` ABC.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.input import Normalize
from gpytorch.means import Mean
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood

# The UQ quality a model's posterior actually provides. This is a property of the
# surrogate, so it lives on the constructor that builds it (NOT a string the caller
# passes alongside — that could silently disagree with the real model). The loop
# checks ``policy.uq_requirement`` against this at fit time (see
# ``session.check_uq_pairing``). Ordering (weakest->strongest): a point estimate, a
# full set of posterior samples, calibrated variance.
UQCapability = str  # "point" | "samples" | "calibrated_variance"


@runtime_checkable
class ModelConstructor(Protocol):
    """Build a fitted botorch ``Model`` from training data.

    X:    (n, d) design inputs
    Y:    (n, m) observed outputs (m=1 for scalar objectives)
    Yvar: (n, m) optional observation-noise variances (-> train_Yvar). When None
          the model infers the noise level (e.g. a score from one noisy observation).

    ``provides_uq`` declares the UQ quality the built model's posterior supplies, so
    the loop can refuse a policy that needs more than the surrogate can give.
    """

    provides_uq: UQCapability

    def build(
        self, X: torch.Tensor, Y: torch.Tensor, Yvar: torch.Tensor | None = None
    ) -> Model: ...


Kernel = Literal["default", "matern52"]

# A prior mean as the GP sees it: raw inputs (..., n, d) -> one output's values
# (..., n) in maximize-canonical space. ``default_constructor(spec, prior_mean=...)``
# builds these from named functions in raw units.
PriorMean = Callable[[torch.Tensor], torch.Tensor]


class _PriorMean(Mean):
    """A GP mean that follows a prior model: ``scale * prior(x) + offset``, in the
    GP's standardized space. ``offset`` is learned with the other settings, and
    ``scale`` is 1 unless ``fit_scale``. The GP then models how the data differ from
    the prior, and its uncertainty is that difference's (the prior's own isn't
    known). Like Xopt's ``CustomMean``, it maps the GP's transformed inputs back to
    raw units and the prior's values into the GP's scaled outputs."""

    def __init__(self, prior: PriorMean, input_transform, y_mean, y_std, fit_scale: bool) -> None:
        super().__init__()
        self.prior = prior
        self._untransform = (input_transform,)  # kept out of the module tree
        self.register_buffer("y_mean", y_mean.reshape(()).detach().clone())
        self.register_buffer("y_std", y_std.reshape(()).detach().clone())
        self.offset = torch.nn.Parameter(torch.zeros((), dtype=torch.double))
        if fit_scale:
            self.scale = torch.nn.Parameter(torch.ones((), dtype=torch.double))
        else:
            self.register_buffer("scale", torch.ones((), dtype=torch.double))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transform = self._untransform[0]
        raw = transform.untransform(x) if transform is not None else x
        value = torch.as_tensor(self.prior(raw), dtype=x.dtype).expand(x.shape[:-1])
        return self.scale * (value - self.y_mean) / self.y_std + self.offset


def _covar_module(kernel: Kernel, d: int):
    """``None`` keeps BoTorch's default kernel; ``"matern52"`` is a scaled ARD
    Matern-5/2 (the sklearn ``ConstantKernel * Matern(nu=2.5)`` multibax used)."""
    if kernel == "default":
        return None
    if kernel == "matern52":
        from gpytorch.kernels import MaternKernel, ScaleKernel

        return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
    raise ValueError(f"kernel must be 'default' or 'matern52', got {kernel!r}")


def _input_transform(d: int, bounds: torch.Tensor | None, log: list[int] | None,
                     indices: list[int] | None = None):
    """Inputs to the unit cube: ``Normalize`` to ``bounds`` (from the data when
    None), after a log10 of the ``log`` columns. ``indices`` limits normalization
    to some columns (the rest, e.g. a task label, pass through)."""
    if not log:
        if bounds is None:
            return Normalize(d=d, indices=indices)
        return Normalize(d=d, indices=indices, bounds=bounds.double())
    from botorch.models.transforms.input import ChainedInputTransform, Log10

    if bounds is not None:
        bounds = bounds.double().clone()
        bounds[:, log] = bounds[:, log].log10()
    normalize = (
        Normalize(d=d, indices=indices)
        if bounds is None
        else Normalize(d=d, indices=indices, bounds=bounds)
    )
    return ChainedInputTransform(log=Log10(indices=list(log)), normalize=normalize)


def _cont_kernel_factory(kernel: Kernel):
    """``MixedSingleTaskGP``'s kernel over the numeric inputs, for ``kernel``."""
    if kernel == "default":
        return None
    from gpytorch.kernels import MaternKernel

    return lambda batch_shape, ard_num_dims, active_dims: MaternKernel(
        nu=2.5, batch_shape=batch_shape, ard_num_dims=ard_num_dims, active_dims=active_dims
    )


def _gp(X, Y, Yvar, bounds, log, kernel: Kernel, categorical: list[int],
        prior: PriorMean | None = None, fit_scale: bool = False) -> SingleTaskGP:
    """One unfitted GP on ``(X, Y)``: a ``SingleTaskGP``, or with ``categorical``
    columns (codes of labeled values) a ``MixedSingleTaskGP``, whose categorical
    kernel learns how alike each pair of labels is instead of assuming an order.
    Codes skip normalization; the kernel only asks whether two are equal. With a
    ``prior``, its mean follows the prior (see :class:`_PriorMean`)."""
    d = X.shape[-1]
    Yvar = Yvar.double() if Yvar is not None else None
    if not categorical:
        gp = SingleTaskGP(
            X, Y, train_Yvar=Yvar, input_transform=_input_transform(d, bounds, log),
            covar_module=_covar_module(kernel, d),
        )
    else:
        from botorch.models import MixedSingleTaskGP

        numeric = [j for j in range(d) if j not in categorical]
        gp = MixedSingleTaskGP(
            X, Y, cat_dims=list(categorical), train_Yvar=Yvar,
            cont_kernel_factory=_cont_kernel_factory(kernel),
            input_transform=_input_transform(d, bounds, log, indices=numeric) if numeric else None,
        )
    if prior is not None:
        gp.mean_module = _PriorMean(
            prior, getattr(gp, "input_transform", None),
            gp.outcome_transform.means, gp.outcome_transform.stdvs, fit_scale,
        )
    return gp


class SingleTaskGPConstructor:
    """Scalar / independent-output GP with input normalization and MLL fit.

    The GP infers the noise level unless Yvar is supplied, and inputs are
    normalized to the problem bounds, after a log of the ``log`` columns (the
    spec's ``log=True`` variables; see ``ProblemSpec.log_indices``).
    ``categorical`` columns hold the codes of labeled values (see
    ``ProblemSpec.categorical_indices``) and get a categorical kernel. A GP
    posterior is a full Gaussian, so it provides calibrated variance (the
    strongest UQ tier).

    ``prior_mean`` makes the GP start from a prior model instead of a constant: a
    function of the raw inputs ``(..., n, d) -> (..., n)`` in maximize-canonical
    space. ``default_constructor(spec, prior_mean=...)`` builds it from a named
    function in raw units, which is the usual way in.
    """

    provides_uq = "calibrated_variance"

    def __init__(
        self,
        bounds: torch.Tensor | None = None,
        kernel: Kernel = "default",
        *,
        log: list[int] | None = None,
        categorical: list[int] | None = None,
        prior_mean: PriorMean | list[PriorMean | None] | None = None,
        fit_prior_scale: bool = False,
    ) -> None:
        # bounds: (2, d) for input Normalize; None -> normalize from data.
        self.bounds = bounds
        self.kernel = kernel
        self.log = list(log or [])
        self.categorical = list(categorical or [])
        self.prior_mean = prior_mean
        self.fit_prior_scale = fit_prior_scale
        _covar_module(kernel, 1)  # validate early

    def build(
        self, X: torch.Tensor, Y: torch.Tensor, Yvar: torch.Tensor | None = None
    ) -> SingleTaskGP:
        prior = self.prior_mean[0] if isinstance(self.prior_mean, list) else self.prior_mean
        gp = _gp(X.double(), Y.double(), Yvar, self.bounds, self.log, self.kernel,
                 self.categorical, prior, self.fit_prior_scale)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        return gp


class ModelListConstructor:
    """Independent GP per output -> ``ModelListGP`` (multi-output / multibax MGPR).

    Each column of Y gets its own ``SingleTaskGP``. This is the rewrite target for
    multibax's independent-sklearn-GP ``MGPR`` and the base for multi-output θ
    models in the SBI app. Each GP posterior is Gaussian -> calibrated variance.
    ``log`` and ``categorical`` as for :class:`SingleTaskGPConstructor`;
    ``prior_mean`` is a list with one entry per output (``None`` for none).
    """

    provides_uq = "calibrated_variance"

    def __init__(
        self,
        bounds: torch.Tensor | None = None,
        kernel: Kernel = "default",
        *,
        log: list[int] | None = None,
        categorical: list[int] | None = None,
        prior_mean: PriorMean | list[PriorMean | None] | None = None,
        fit_prior_scale: bool = False,
    ) -> None:
        self.bounds = bounds
        self.kernel = kernel
        self.log = list(log or [])
        self.categorical = list(categorical or [])
        self.prior_mean = prior_mean
        self.fit_prior_scale = fit_prior_scale
        _covar_module(kernel, 1)  # validate early

    def build(
        self, X: torch.Tensor, Y: torch.Tensor, Yvar: torch.Tensor | None = None
    ) -> ModelListGP:
        X = X.double()
        Y = Y.double()
        priors = self.prior_mean or [None] * Y.shape[-1]
        if callable(priors):
            raise TypeError("ModelListConstructor: prior_mean is a list, one entry per output")
        gps = [
            _gp(
                X, Y[..., j : j + 1], Yvar[..., j : j + 1] if Yvar is not None else None,
                self.bounds, self.log, self.kernel, self.categorical,
                priors[j], self.fit_prior_scale,
            )
            for j in range(Y.shape[-1])
        ]
        model = ModelListGP(*gps)
        fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
        return model


class EnsembleConstructor:
    """A deep-ensemble surrogate -> botorch ``EnsembleModel`` (the non-GP proof).

    Builds ``n_members`` small MLPs and trains each (independent init + a short Adam
    fit) so the spread across members is the uncertainty. This is the generality
    PROOF that the ``ModelConstructor`` seam is not GP-coupled — NOT nn-bax
    productionization (MC-dropout, deep kernels, calibration are F-tier).

    ``provides_uq`` is honest and load-bearing (it feeds ``check_uq_pairing``):
      - ``n_members > 1`` -> ``"samples"`` (coherent member draws; the InfoBAX/
        SwitchBAX contract), which also exposes a (member-spread) variance.
      - ``n_members == 1`` -> ``"point"`` (a single deterministic net; only MeanBAX/
        greedy may pair with it).
    An ensemble's spread is NOT calibrated variance, so it never claims that tier —
    pairing one with ``UncertaintySampling`` is correctly refused.

    ``Yvar`` is ignored: an ensemble represents noise through member disagreement,
    not an explicit per-point variance.
    """

    def __init__(
        self,
        n_members: int = 5,
        *,
        width: int = 16,
        epochs: int = 200,
        lr: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.n_members = n_members
        self.width = width
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.provides_uq = "samples" if n_members > 1 else "point"

    def build(
        self, X: torch.Tensor, Y: torch.Tensor, Yvar: torch.Tensor | None = None
    ):
        from abbot.models.ensemble import ToyEnsembleModel, make_toy_mlp

        X = X.double()
        Y = Y.double()
        d, m = X.shape[-1], Y.shape[-1]
        g = torch.Generator().manual_seed(self.seed)
        members = []
        for _ in range(self.n_members):
            # Independent init per member: reseed torch's global RNG from our
            # generator (avoids Date/random bans elsewhere; deterministic by seed).
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
            net = make_toy_mlp(d, m, width=self.width)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(X), Y)
                loss.backward()
                opt.step()
            members.append(net)
        return ToyEnsembleModel(members, num_outputs=m)


class MultiFidelityGPConstructor:
    """A GP over designs and sources, so cheap runs inform the target: the model for
    a spec with source variables.

    - **Ordered sources** (fidelity knobs): BoTorch's ``SingleTaskMultiFidelityGP``.
      Each source is mapped to [0, 1] with its accurate end (``Source.accurate``) at
      1, on a log scale for ``log=True`` sources. The default linear-truncated kernel
      models a run as the accurate result plus a bias that vanishes at the accurate
      end and varies with the design. ``power`` (a whole number, default 1) shapes how
      that bias fades between the ends; it is fixed rather than learned, because
      BoTorch's learned power is a fraction in general, and the kernel is only valid
      (positive semi-definite) for whole numbers: a fitted power near 0.5 gives
      negative variances. ``linear_truncated=False`` instead multiplies
      the design kernel by a per-source kernel, with ``iteration`` naming a source
      whose results settle towards a limit along it (an exponential-decay kernel;
      iteration counts, epochs).
    - **One unordered source** (different codes, simulation vs experiment): a
      ``MultiTaskGP`` that learns how the sources correlate. The noise level is
      shared across sources unless ``Yvar`` is given. (BoTorch's per-source noise
      likelihood fails in conditioning and in entropy search as of 0.18.)

    Ordered and unordered sources together, or several unordered sources, aren't
    supported yet. A multi-output spec gets one model per output where needed.
    """

    provides_uq = "calibrated_variance"

    def __init__(
        self,
        spec,
        *,
        linear_truncated: bool = True,
        power: int = 1,
        iteration: str | None = None,
    ) -> None:
        if isinstance(power, bool) or not float(power).is_integer() or power < 1:
            raise ValueError(
                f"power must be a positive whole number, got {power!r}: the "
                "linear-truncated kernel is only valid for whole-number powers"
            )
        sources = list(spec.sources or [])
        if not sources:
            raise ValueError("MultiFidelityGPConstructor needs a spec with sources")
        ordered = [s for s in sources if s.is_ordered]
        unordered = [s for s in sources if not s.is_ordered]
        if ordered and unordered:
            raise ValueError(
                "MultiFidelityGPConstructor doesn't yet model ordered and unordered "
                f"sources together (ordered: {[s.name for s in ordered]}, unordered: "
                f"{[s.name for s in unordered]})"
            )
        if len(unordered) > 1:
            raise ValueError(
                "MultiFidelityGPConstructor models one unordered source; got "
                f"{[s.name for s in unordered]}"
            )
        if iteration is not None and iteration not in [s.name for s in ordered]:
            raise ValueError(
                f"iteration={iteration!r} is not an ordered source "
                f"({[s.name for s in ordered]})"
            )
        self.spec = spec
        self.linear_truncated = linear_truncated
        self.iteration = iteration
        self.power = int(power)
        self.bounds = spec.bounds_tensor()

    def build(self, X: torch.Tensor, Y: torch.Tensor, Yvar: torch.Tensor | None = None):
        X, Y = X.double(), Y.double()
        Yvar = Yvar.double() if Yvar is not None else None
        if all(s.is_ordered for s in self.spec.sources):
            return self._fidelity_gp(X, Y, Yvar)
        return self._multitask_gp(X, Y, Yvar)

    def _fidelity_gp(self, X, Y, Yvar):
        from botorch.models import SingleTaskMultiFidelityGP

        spec = self.spec
        positions = dict(zip(spec.source_names, spec.source_indices()))
        iteration = positions[self.iteration] if self.iteration is not None else None
        gp = SingleTaskMultiFidelityGP(
            X,
            Y,
            train_Yvar=Yvar,
            iteration_fidelity=iteration,
            data_fidelities=[i for i in positions.values() if i != iteration] or None,
            linear_truncated=self.linear_truncated,
            input_transform=_fidelity_transform(spec, self.bounds),
        )
        if spec.categorical_indices():
            _add_categories(gp, spec.categorical_indices(), iteration, positions.values(),
                            self.linear_truncated)
        _pin_linear_truncated_power(gp, self.power)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        return gp

    def _multitask_gp(self, X, Y, Yvar):
        from botorch.models import MultiTaskGP
        from botorch.models.utils.gpytorch_modules import (
            get_gaussian_likelihood_with_lognormal_prior,
        )

        (source,) = self.spec.sources
        (t,) = self.spec.source_indices()
        d = X.shape[-1]
        categorical = self.spec.categorical_indices()
        design = [i for i in range(d) if i != t and i not in categorical]
        gps = []
        for j in range(Y.shape[-1]):
            gp = MultiTaskGP(
                X,
                Y[..., j : j + 1],
                train_Yvar=Yvar[..., j : j + 1] if Yvar is not None else None,
                task_feature=t,
                output_tasks=[int(source.target_code)],
                all_tasks=sorted(int(v) for v in source.domain.codes),
                likelihood=get_gaussian_likelihood_with_lognormal_prior()
                if Yvar is None
                else None,
                input_transform=_input_transform(
                    d, self.bounds, self.spec.log_indices(), indices=design
                ) if design else None,
                covar_module=_design_kernel(design, categorical, d) if categorical else None,
            )
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            gps.append(gp)
        return gps[0] if len(gps) == 1 else ModelListGP(*gps)


def _design_kernel(numeric: list[int], categorical: list[int], d: int):
    """A ``MultiTaskGP`` data kernel for designs with labeled variables: BoTorch's
    default kernel on the ``numeric`` columns times a categorical kernel on the
    ``categorical`` ones, indexed into the full ``d`` columns."""
    from botorch.models.kernels.categorical import CategoricalKernel
    from botorch.models.utils.gpytorch_modules import (
        get_covar_module_with_dim_scaled_prior,
    )
    from gpytorch.kernels import ScaleKernel

    cat = CategoricalKernel(ard_num_dims=len(categorical), active_dims=categorical)
    if not numeric:
        kernel = ScaleKernel(cat)
    else:
        kernel = ScaleKernel(
            get_covar_module_with_dim_scaled_prior(ard_num_dims=len(numeric), active_dims=numeric)
            * cat
        )
    kernel.active_dims = torch.arange(d)  # columns are absolute; MultiTaskGP keeps a set one
    return kernel


def _add_categories(gp, categorical: list[int], iteration, data_fidelities, linear_truncated) -> None:
    """Give a ``SingleTaskMultiFidelityGP`` a categorical kernel on the
    ``categorical`` columns (codes of labeled values): its fidelity kernel is
    rebuilt on the other columns, times a categorical kernel on these. The
    per-output parameter map (``_subset_batch_dict``) is rebuilt to match, so
    ``subset_output`` still works."""
    from botorch.models.gp_regression_fidelity import _setup_multifidelity_covar_module
    from botorch.models.kernels.categorical import CategoricalKernel
    from gpytorch.kernels import ScaleKernel

    d = gp.train_inputs[0].shape[-1]
    keep = [i for i in range(d) if i not in categorical]
    at = {j: i for i, j in enumerate(keep)}
    data = [at[i] for i in data_fidelities if i != iteration]
    covar, subset = _setup_multifidelity_covar_module(
        dim=len(keep),
        aug_batch_shape=gp._aug_batch_shape,
        iteration_fidelity=at[iteration] if iteration is not None else None,
        data_fidelities=data or None,
        linear_truncated=linear_truncated,
        nu=2.5,
    )
    product, prefix = (
        (covar.base_kernel, "covar_module.base_kernel.kernels")
        if isinstance(covar, ScaleKernel)
        else (covar, "covar_module.kernels")
    )
    keep_t = torch.tensor(keep)
    for k in product.kernels:  # their columns are positions among ``keep``
        k.active_dims = keep_t if k.active_dims is None else keep_t[k.active_dims]
    product.kernels.append(
        CategoricalKernel(
            batch_shape=gp._aug_batch_shape, ard_num_dims=len(categorical), active_dims=categorical
        )
    )
    subset[f"{prefix}.{len(product.kernels) - 1}.raw_lengthscale"] = -3
    if isinstance(covar, ScaleKernel):
        subset["covar_module.raw_outputscale"] = -1
    gp.covar_module = covar.to(gp.train_inputs[0])
    gp._subset_batch_dict = {
        **{k: v for k, v in gp._subset_batch_dict.items() if not k.startswith("covar_module")},
        **subset,
    }


def _pin_linear_truncated_power(gp, power: int) -> None:
    """Fix each linear-truncated kernel's polynomial power at ``power``.

    BoTorch learns it as a continuous parameter, but the bias term
    ``(1 + s s')^p`` is only a valid (positive semi-definite) kernel for whole
    numbers ``p``. A fitted ``p`` near 0.5 gives negative posterior variances, which
    turn into NaNs in entropy search.
    """
    from botorch.models.kernels import LinearTruncatedFidelityKernel

    for kernel in gp.covar_module.modules():
        if isinstance(kernel, LinearTruncatedFidelityKernel):
            kernel.power = float(power)
            kernel.raw_power.requires_grad_(False)


def _fidelity_transform(spec, bounds: torch.Tensor):
    """Inputs to the unit cube, with each ordered source's accurate end at 1, as
    BoTorch's fidelity kernels expect. ``log=True`` inputs (sources and design
    variables) are on a log scale."""
    from botorch.models.transforms.input import (
        AffineInputTransform,
        ChainedInputTransform,
        Log10,
    )

    lower, upper = bounds.double().clone()
    flip = torch.zeros_like(lower, dtype=torch.bool)
    logged = spec.log_indices()
    for i in logged:
        lower[i], upper[i] = lower[i].log10(), upper[i].log10()
    for i, source in zip(spec.source_indices(), spec.sources):
        flip[i] = source.accurate == "low"
    span = (upper - lower).where(upper > lower, torch.ones_like(lower))
    # A hair wider than the range, so round-off (e.g. in the log) never puts a source
    # above 1, where the product kernel's (1 - s)^power is NaN.
    span = span * (1 + 1e-9)
    # (x - offset) / coefficient: lower -> 0 and upper -> 1, or reversed when flipped.
    affine = AffineInputTransform(
        d=len(lower),
        coefficient=torch.where(flip, -span, span),
        offset=torch.where(flip, upper, lower),
    )
    if not logged:
        return affine
    return ChainedInputTransform(log=Log10(indices=logged), scale=affine)


def default_constructor(
    spec,
    bounds: torch.Tensor | None = None,
    *,
    prior_mean: dict[str, Callable] | None = None,
    fit_prior_scale: bool = False,
) -> ModelConstructor:
    """Pick a sensible builtin: single-output GP for one OUTPUT, else ModelList, and
    a :class:`MultiFidelityGPConstructor` when the spec has sources.

    Keyed on ``n_outputs`` (every output is modeled, incl. OBSERVE), not objective
    count — a single-objective spec with an extra OBSERVE output needs a ModelList.
    Callers may still pass their own ``ModelConstructor`` (e.g. a correlated
    MultiTaskGP for SBI). ``bounds`` defaults to the spec's design bounds,
    ``log=True`` variables are modeled on a log scale, and labeled variables get a
    categorical kernel.

    ``prior_mean={output name: fn}`` starts an output's GP from a prior model (a
    physics model, a fit to an earlier dataset) instead of a constant; the GP learns
    how the data differ from it. ``fn`` takes ``{variable name: tensor}`` in raw units
    (labeled variables as codes) and returns the output's raw values, as torch
    operations: proposals are optimized by gradient. Never pre-negate a minimized
    output. A constant offset from the prior is learned; ``fit_prior_scale=True``
    also learns a scale, so data can shrink a prior that's off in amplitude.
    """
    if spec.sources:
        if prior_mean:
            raise ValueError(
                "prior_mean isn't supported with source variables yet "
                f"({spec.source_names})"
            )
        return MultiFidelityGPConstructor(spec)
    if bounds is None:
        bounds = spec.bounds_tensor()
    kw = {"log": spec.log_indices(), "categorical": spec.categorical_indices()}
    if prior_mean:
        priors = _priors(spec, prior_mean)
        kw.update(prior_mean=priors if spec.n_outputs > 1 else priors[0],
                  fit_prior_scale=fit_prior_scale)
    if spec.n_outputs == 1:
        return SingleTaskGPConstructor(bounds=bounds, **kw)
    return ModelListConstructor(bounds=bounds, **kw)


def _priors(spec, prior_mean: dict[str, Callable]) -> list[PriorMean | None]:
    """Named raw-unit prior functions as the GPs take them, one per output, each
    checked on a few designs so a broken one fails here rather than in a fit."""
    unknown = sorted(set(prior_mean) - set(spec.output_names))
    if unknown:
        raise ValueError(f"prior_mean: no output named {unknown}; the outputs are {spec.output_names}")
    names = spec.variable_names
    signs = spec.output_signs().tolist()

    def canonical(fn, sign):
        def prior(X: torch.Tensor) -> torch.Tensor:
            return sign * torch.as_tensor(fn({n: X[..., k] for k, n in enumerate(names)}),
                                          dtype=X.dtype)
        return prior

    out: list[PriorMean | None] = []
    probe = spec.sample_designs(4)
    for name, sign in zip(spec.output_names, signs):
        fn = prior_mean.get(name)
        if fn is None:
            out.append(None)
            continue
        prior = canonical(fn, sign)
        try:
            value = torch.as_tensor(prior(probe)).expand(probe.shape[:-1])
        except Exception as exc:
            raise ValueError(
                f"prior_mean for {name!r} failed on sample designs: {type(exc).__name__}: "
                f"{exc}. It takes {{variable name: tensor}} and returns a tensor of values."
            ) from exc
        if not torch.isfinite(value).all():
            raise ValueError(f"prior_mean for {name!r} returned non-finite values")
        out.append(prior)
    return out
