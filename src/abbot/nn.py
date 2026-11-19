"""Neural-network surrogates (experimental).

Bring your own network; abbot does the rest. ``make_net(d_in, d_out)`` returns a
``torch.nn.Module`` mapping a batch of inputs ``(N, d_in)`` to ``(N, d_out)``.
abbot scales the inputs to the unit box (or hands the raw designs to your
``featurize``), standardizes the outputs, trains, warm-starts the next fit from the
last one, skips outputs a run didn't measure, and wraps the result as a BoTorch
model, so every policy reads it like any other.

Three ways to get uncertainty from a network:

- :class:`NetEnsembleConstructor`: independently initialized copies of the network
  (a deep ensemble). A posterior sample is one member.
- :class:`MCDropoutConstructor`: one network with ``torch.nn.Dropout`` layers,
  sampled with dropout on. A posterior sample is one fixed dropout mask, shared by
  every design it's evaluated at, so a sample is one coherent function (with a
  fresh mask per design, the "sample" would be independent noise at each design
  and the goal would run on nonsense).
- :class:`DeepKernelConstructor`: the network maps designs to features and a GP
  works on the features (deep kernel learning). It's a GP, so it gives calibrated
  variance and exact conditioning, and every policy works with it.

InfoBAX needs the model updated as if it had seen a sample's values at the
target set. A GP does that exactly. The ensemble and dropout models do it by
fine-tuning a copy on the data plus those values (``refit_epochs``), as nn-bax
does. By default the entropy before the update comes from a matching fine-tune on
the data alone (``control=True``), so the change from fine-tuning itself (members
drifting together as they train on) isn't counted as information.

Entropies are Gaussian, from each design's predicted variance. That's exact for a
GP; for an ensemble or dropout it treats the spread of the samples as Gaussian.

This module is experimental: its interface may change between releases.
"""

from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Callable
from typing import Any

import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.ensemble import EnsembleModel
from botorch.models.transforms.outcome import Standardize
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from abbot.model import measured

logger = logging.getLogger(__name__)

MakeNet = Callable[[int, int], torch.nn.Module]
# ``train(net, X, Y, epochs)``: fit ``net`` (raw designs ``(n, d)`` -> standardized
# outputs ``(n, m)``) to ``Y`` (standardized; NaN where a run didn't measure an
# output) for ``epochs`` passes. Replaces the default (Adam on the squared error).
Train = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor, int], None]


# --- the network as abbot calls it ---------------------------------------------


class _Net(torch.nn.Module):
    """A user network behind abbot's input handling: designs ``(..., n, d)`` in raw
    units in, ``(..., n, m)`` out. Inputs are scaled to the unit box (``bounds``,
    else the training data's range), or passed raw to ``featurize``."""

    def __init__(self, net, d: int, m: int, lo, hi, featurize=None) -> None:
        super().__init__()
        self.net = net
        self.d, self.m = d, m
        self.featurize = featurize
        self.register_buffer("lo", lo)
        self.register_buffer("span", (hi - lo).clamp_min(1e-12))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        lead = X.shape[:-1]
        flat = X.reshape(-1, X.shape[-1])
        inputs = self.featurize(flat) if self.featurize is not None else (flat - self.lo) / self.span
        out = self.net(inputs)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        return out.reshape(*lead, out.shape[-1])


def _box(X: torch.Tensor, bounds) -> tuple[torch.Tensor, torch.Tensor]:
    if bounds is not None:
        b = torch.as_tensor(bounds, dtype=X.dtype, device=X.device)
        return b[0], b[1]
    return X.min(dim=0).values, X.max(dim=0).values


def adam(
    *, lr: float = 1e-2, weight_decay: float = 0.0, batch_size: int | None = None
) -> Train:
    """The default trainer: Adam on the mean squared error over the measured
    outputs, full batch unless ``batch_size`` is given."""

    def train(net, X, Y, epochs):
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        seen = ~Y.isnan()
        target = Y.nan_to_num()
        n = len(X)
        size = n if batch_size is None else min(batch_size, n)
        net.train()
        for _ in range(epochs):
            order = torch.randperm(n, device=X.device) if size < n else None
            for start in range(0, n, size):
                rows = order[start : start + size] if order is not None else slice(None)
                mask = seen[rows]
                if not mask.any():
                    continue
                opt.zero_grad()
                loss = ((net(X[rows]) - target[rows]) ** 2)[mask].mean()
                loss.backward()
                opt.step()
        net.eval()

    return train


# --- ensemble and dropout: a set of functions ---------------------------------


class _NetSamplesModel(EnsembleModel):
    """A BoTorch ``EnsembleModel`` whose samples are network functions (ensemble
    members, or dropout masks), in the campaign's (canonical) output units.

    ``condition_on_observations`` fine-tunes a copy on the training data plus the
    new points; ``refit_control`` fine-tunes one on the data alone, for InfoBAX's
    entropy before the update.
    """

    def __init__(self, nets, X, Y, *, y_mean, y_std, noise, train, refit_epochs, control, seeds=None):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self._num_outputs = Y.shape[-1]
        self.train_X, self.train_Y = X, Y  # raw canonical, NaN where unmeasured
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)
        self.register_buffer("noise", noise)  # (m,) standardized noise variance
        self._train = train
        self.refit_epochs = refit_epochs
        self.control = control
        self.seeds = seeds  # dropout: one seed per sample; None: an ensemble
        self._control_model = None

    @property
    def num_samples(self) -> int:
        return len(self.seeds) if self.seeds is not None else len(self.nets)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.seeds is None:
            values = torch.stack([net(X) for net in self.nets], dim=-3)
        else:
            net = self.nets[0]
            values = torch.stack([_with_mask(net, seed, X) for seed in self.seeds], dim=-3)
        return values * self.y_std + self.y_mean

    def observation_noise_variance(self, X: torch.Tensor) -> torch.Tensor:
        """Measurement-noise variance ``(N, m)`` in output units at designs ``X``."""
        noise = self.noise * self.y_std**2
        return noise.expand(X.reshape(-1, X.shape[-1]).shape[0], -1)

    def _fine_tuned(self, X: torch.Tensor, Y: torch.Tensor):
        twin = copy.deepcopy(self)
        twin._control_model = None
        Z = (Y - self.y_mean) / self.y_std
        with torch.enable_grad():
            for net in twin.nets:
                twin._train(net, X, Z, self.refit_epochs)
        twin.train_X, twin.train_Y = X, Y
        return twin

    def condition_on_observations(self, X: torch.Tensor, Y: torch.Tensor, noise=None, **kwargs: Any):
        """A copy fine-tuned (``refit_epochs``) on the training data plus ``(X, Y)``,
        ``Y`` in output units. The networks fit the new points; ``noise`` is
        ignored."""
        if X.dim() != 2:
            raise NotImplementedError("a network model conditions on one set of points at a time")
        X_all = torch.cat([self.train_X, X.to(self.train_X)])
        Y_all = torch.cat([self.train_Y, Y.reshape(len(X), -1).to(self.train_Y)])
        return self._fine_tuned(X_all, Y_all)

    def refit_control(self):
        """A copy fine-tuned on the training data alone, as long as a conditioned
        copy is: InfoBAX's entropy before the update. ``None`` with ``control=False``.
        """
        if not self.control:
            return None
        if self._control_model is None:
            self._control_model = self._fine_tuned(self.train_X, self.train_Y)
        return self._control_model


def _with_mask(net: torch.nn.Module, seed: int, X: torch.Tensor) -> torch.Tensor:
    """``net(X)`` with each ``Dropout`` layer applying one mask, drawn from ``seed``,
    to every design: the same function wherever it's evaluated. The mask is over
    the layer's last dimension (the features)."""
    dropouts = [m for m in net.modules() if isinstance(m, torch.nn.Dropout)]
    hooks = []

    def hook_for(i):
        def hook(module, _inputs, out):
            if module.p == 0:
                return out
            g = torch.Generator(device=out.device).manual_seed(seed * 7919 + i)
            keep = torch.rand(out.shape[-1], generator=g, device=out.device, dtype=out.dtype)
            return out * (keep >= module.p).to(out.dtype) / (1 - module.p)

        return hook

    for i, module in enumerate(dropouts):
        hooks.append(module.register_forward_hook(hook_for(i)))
    try:
        net.eval()  # the layers pass through; the hooks apply the masks
        return net(X)
    finally:
        for h in hooks:
            h.remove()


class _NetConstructor:
    """Shared by the ensemble and dropout constructors."""

    provides_uq = "samples"
    handles_missing = True  # an output a run didn't measure is left out of the loss
    supports_conditioning = True  # by fine-tuning

    def __init__(
        self,
        make_net: MakeNet,
        *,
        bounds=None,
        featurize: Callable[[torch.Tensor], Any] | None = None,
        epochs: int = 500,
        warm_epochs: int | None = 200,
        refit_epochs: int = 100,
        control: bool = True,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
        train: Train | None = None,
        noise: float | list[float] | None = None,
        min_noise: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.make_net = make_net
        self.bounds = bounds
        self.featurize = featurize
        self.epochs = epochs
        self.warm_epochs = warm_epochs
        self.refit_epochs = refit_epochs
        self.control = control
        self.train = train or adam(lr=lr, weight_decay=weight_decay, batch_size=batch_size)
        self.noise = noise
        self.min_noise = min_noise
        self.seed = seed
        self._last: list[dict] | None = None  # trained weights, to warm-start the next fit

    def _make(self, i: int, X, m: int) -> _Net:
        with torch.random.fork_rng(devices=[X.device] if X.device.type == "cuda" else []):
            torch.manual_seed(self.seed * 1000 + i)
            net = self.make_net(X.shape[-1], m)
        lo, hi = _box(X, self.bounds)
        return _Net(net, X.shape[-1], m, lo, hi, self.featurize).to(X)

    def _n_nets(self) -> int:
        raise NotImplementedError

    def _fit_nets(self, X, Z):
        m = Z.shape[-1]
        nets = [self._make(i, X, m) for i in range(self._n_nets())]
        warm = self.warm_epochs is not None and self._last is not None
        if warm:
            try:
                for net, state in zip(nets, self._last, strict=True):
                    net.net.load_state_dict(state)
            except (RuntimeError, ValueError) as exc:  # the network changed shape
                logger.info("Not warm-starting the network (%s).", exc)
                nets = [self._make(i, X, m) for i in range(self._n_nets())]
                warm = False
        epochs = self.warm_epochs if warm else self.epochs
        for net in nets:
            self.train(net, X, Z, epochs)
        self._last = [copy.deepcopy(net.net.state_dict()) for net in nets]
        return nets

    def _noise(self, nets, X, Z, Yvar, y_std):
        """Standardized measurement-noise variance per output: given (``noise``, in
        output units), the mean ``Yvar``, else the networks' mean squared residual,
        at least ``min_noise``."""
        m = Z.shape[-1]
        if self.noise is not None:
            raw = torch.as_tensor(self.noise, dtype=Z.dtype, device=Z.device).expand(m)
            return raw / y_std**2
        if Yvar is not None:
            return (Yvar.nanmean(dim=0) / y_std**2).nan_to_num(self.min_noise).clamp_min(self.min_noise)
        with torch.no_grad():
            pred = torch.stack([self._predict(nets, i, X) for i in range(self._n_samples(nets))]).mean(0)
        resid = (pred - Z) ** 2
        return resid.nanmean(dim=0).nan_to_num(self.min_noise).clamp_min(self.min_noise)

    def _predict(self, nets, i, X):
        return nets[i](X)

    def _n_samples(self, nets) -> int:
        return len(nets)

    def _seeds(self):
        return None

    def build(self, X: torch.Tensor, Y: torch.Tensor, Yvar: torch.Tensor | None = None):
        X, Y = X.double(), Y.double()
        y_mean = Y.nanmean(dim=0).nan_to_num(0.0)
        y_std = _nanstd(Y).nan_to_num(1.0).clamp_min(1e-9)
        y_std = torch.where(y_std > 1e-9, y_std, torch.ones_like(y_std))
        Z = (Y - y_mean) / y_std
        with torch.enable_grad():
            nets = self._fit_nets(X, Z)
        noise = self._noise(nets, X, Z, Yvar, y_std)
        return _NetSamplesModel(
            nets, X, Y, y_mean=y_mean, y_std=y_std, noise=noise, train=self.train,
            refit_epochs=self.refit_epochs, control=self.control, seeds=self._seeds(),
        )


def _nanstd(Y: torch.Tensor) -> torch.Tensor:
    mean = Y.nanmean(dim=0)
    count = (~Y.isnan()).sum(dim=0)
    var = ((Y - mean) ** 2).nansum(dim=0) / (count - 1).clamp_min(1)
    return var.sqrt()


class NetEnsembleConstructor(_NetConstructor):
    """A deep ensemble of your network (experimental): ``n_members`` copies, each
    from its own initial weights, trained on the same data. A posterior sample is
    one member; the spread across members is the uncertainty.

    ``make_net(d_in, d_out)`` builds one member (``d_in`` is the number of design
    columns; with ``featurize``, your network takes whatever ``featurize`` returns).
    It's converted to float64 and moved to the campaign's device.

    - **Inputs:** scaled to the unit box given by ``bounds`` (``(2, d)``; pass
      ``spec.bounds_tensor()``), else the training data's range at each fit.
      ``featurize(X)`` instead turns raw designs ``(N, d)`` into the network's input
      (an index variable into a set of structures, say), and skips the scaling.
    - **Training:** ``epochs`` from fresh weights; later fits start from the last
      fit's weights and train ``warm_epochs`` (``None``: always from fresh). The
      default trainer is Adam on the squared error (``lr``, ``weight_decay``,
      ``batch_size``); ``train(net, X, Y, epochs)`` replaces it.
    - **Noise:** ``noise`` (variance in output units), else the mean ``Yvar``, else
      the mean squared residual, at least ``min_noise`` (as a fraction of each
      output's variance).
    - **InfoBAX:** conditioning fine-tunes a copy for ``refit_epochs``;
      ``control`` as in the module docs.

    Provides posterior samples, so MeanBAX, PS-BAX, SwitchBAX and InfoBAX work with
    it; a method that needs calibrated variance (US, the BO policies) is refused.
    """

    def __init__(self, make_net: MakeNet, n_members: int = 5, **kwargs) -> None:
        if n_members < 2:
            raise ValueError("an ensemble needs at least 2 members for a spread")
        super().__init__(make_net, **kwargs)
        self.n_members = n_members

    def _n_nets(self) -> int:
        return self.n_members


class MCDropoutConstructor(_NetConstructor):
    """Your network with dropout on as the uncertainty (experimental). The network
    needs ``torch.nn.Dropout`` layers; it's trained as usual (a fresh mask per
    design and step), and sampled with ``n_samples`` fixed masks, each shared by
    every design (see the module docs). A posterior sample is one mask.

    Options as for :class:`NetEnsembleConstructor`. ``seed`` also fixes the masks.
    """

    def __init__(self, make_net: MakeNet, n_samples: int = 32, **kwargs) -> None:
        if n_samples < 2:
            raise ValueError("dropout needs at least 2 samples for a spread")
        super().__init__(make_net, **kwargs)
        self.n_samples = n_samples

    def _n_nets(self) -> int:
        return 1

    def _seeds(self):
        return list(range(self.seed * 100003, self.seed * 100003 + self.n_samples))

    def _make(self, i, X, m):
        net = super()._make(i, X, m)
        if not any(isinstance(layer, torch.nn.Dropout) for layer in net.modules()):
            raise ValueError(
                "MCDropoutConstructor: the network has no torch.nn.Dropout layer, so "
                "dropout gives no spread; add one, or use NetEnsembleConstructor"
            )
        return net

    def _n_samples(self, nets) -> int:
        return self.n_samples

    def _predict(self, nets, i, X):
        return _with_mask(nets[0], self._seeds()[i], X)


# --- deep kernel: a GP on the network's features ------------------------------


class _DeepKernelGP(SingleTaskGP):
    """A ``SingleTaskGP`` whose kernel works on ``feature_net(x)``, scaled to
    [-1, 1]. Conditioning, fantasies and the posterior are the GP's own."""

    def __init__(self, X, Y, Yvar, feature_net: _Net, n_features: int) -> None:
        from gpytorch.utils.grid import ScaleToBounds

        with warnings.catch_warnings():  # the network scales the inputs, not botorch
            warnings.filterwarnings("ignore", message=".*unit cube.*")
            warnings.filterwarnings("ignore", message=".*standardize.*")
            super().__init__(
                X, Y, train_Yvar=Yvar,
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=n_features)),
                outcome_transform=Standardize(m=1),
            )
        self.feature_net = feature_net
        self.scale_features = ScaleToBounds(-1.0, 1.0)
        self.to(X)

    def forward(self, x):
        if self.training:
            x = self.transform_inputs(x)
        z = self.scale_features(self.feature_net(x))
        return MultivariateNormal(self.mean_module(z), self.covar_module(z))


def _project(gp) -> None:
    """Keep settings whose constraint has no transform (BoTorch's noise, which its
    own fit keeps in bounds with L-BFGS-B) inside their bounds after an Adam step."""
    with torch.no_grad():
        for _name, param, constraint in gp.named_parameters_and_constraints():
            if constraint is not None and not constraint.enforced:
                param.clamp_(
                    min=float(constraint.lower_bound.min()),
                    max=float(constraint.upper_bound.max()),
                )


class DeepKernelConstructor:
    """A GP on features your network learns (deep kernel learning; experimental).

    ``make_net(d_in, n_features)`` maps designs to ``n_features`` features; a GP
    with a Matérn kernel works on them, and the network and the GP's settings are
    trained together on the GP's marginal likelihood (Adam, ``epochs``, ``lr``).
    Each output gets its own network and GP, on the runs that measured it.

    It's a GP, so it provides calibrated variance and exact conditioning: every
    policy works with it, InfoBAX without fine-tuning. With few runs the network
    can overfit, which shows as overconfidence; keep it small, or use
    ``weight_decay``. ``bounds``, ``featurize``, ``warm_epochs`` and ``seed`` as for
    :class:`NetEnsembleConstructor`.
    """

    provides_uq = "calibrated_variance"
    handles_missing = True
    supports_conditioning = True

    def __init__(
        self,
        make_net: MakeNet,
        n_features: int = 4,
        *,
        bounds=None,
        featurize: Callable[[torch.Tensor], Any] | None = None,
        epochs: int = 300,
        warm_epochs: int | None = 100,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.make_net = make_net
        self.n_features = n_features
        self.bounds = bounds
        self.featurize = featurize
        self.epochs = epochs
        self.warm_epochs = warm_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self._last: list[dict] | None = None

    def _one(self, j: int, X, Y, Yvar):
        with torch.random.fork_rng(devices=[X.device] if X.device.type == "cuda" else []):
            torch.manual_seed(self.seed * 1000 + j)
            net = self.make_net(X.shape[-1], self.n_features)
        lo, hi = _box(X, self.bounds)
        feature_net = _Net(net, X.shape[-1], self.n_features, lo, hi, self.featurize).to(X)
        epochs = self.epochs
        if self.warm_epochs is not None and self._last is not None and j < len(self._last):
            try:
                feature_net.net.load_state_dict(self._last[j])
                epochs = self.warm_epochs
            except (RuntimeError, ValueError) as exc:
                logger.info("Not warm-starting the network (%s).", exc)
        gp = _DeepKernelGP(X, Y, Yvar, feature_net, self.n_features)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        opt = torch.optim.Adam(
            [
                {"params": gp.feature_net.parameters(), "weight_decay": self.weight_decay},
                {"params": [p for n, p in gp.named_parameters() if not n.startswith("feature_net.")]},
            ],
            lr=self.lr,
        )
        gp.train()
        with torch.enable_grad():
            for _ in range(epochs):
                opt.zero_grad()
                out = gp(*gp.train_inputs)
                loss = -mll(out, gp.train_targets).sum()
                loss.backward()
                opt.step()
                _project(gp)
        gp.eval()
        return gp

    def build(self, X: torch.Tensor, Y: torch.Tensor, Yvar: torch.Tensor | None = None):
        X, Y = X.double(), Y.double()
        gps = []
        for j in range(Y.shape[-1]):
            Xj, Yj, Vj = measured(X, Y, Yvar, j)
            if not len(Xj):
                raise ValueError(f"no run has measured output {j} yet, so there's nothing to fit it on")
            gps.append(self._one(j, Xj, Yj, Vj))
        self._last = [copy.deepcopy(gp.feature_net.net.state_dict()) for gp in gps]
        return gps[0] if len(gps) == 1 else ModelListGP(*gps)


__all__ = [
    "DeepKernelConstructor",
    "MCDropoutConstructor",
    "NetEnsembleConstructor",
    "adam",
]
