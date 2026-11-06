"""Matplotlib figures of a campaign: the model, the progress, and the model check.

Each function takes a :class:`~abbot.session.Campaign` and returns a
``matplotlib.figure.Figure`` (save it with ``fig.savefig(path)``). Values are in raw
units, as measured. Needs matplotlib (``pip install matplotlib``; the ``mcp`` extra
includes it).

- :func:`model`: what the model predicts. One variable: the mean with a 95% band,
  the runs, the next designs, and the goal (a band or threshold on the output axis,
  and the stretches of the variable the model currently puts in the goal region).
  Two: contours of the mean and its uncertainty. More: one slice per variable
  through the recommended design (or ``at=``), or a contour slice for two named
  ``variables``.
- :func:`progress`: the best so far against runs (one objective), the dominated
  hypervolume (several), or each run's goal outputs against the goal (a goal and no
  objective).
- :func:`check`: each run's leave-one-out error in standard deviations, the numbers
  behind :meth:`Campaign.check`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from abbot import analysis
from abbot.goal import BandIntersection, LevelSet, Wishlist
from abbot.problem import EnumeratedDomain, IntegerDomain

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from abbot.session import Campaign

MEAN = "#4f46e5"
BAND = "#4f46e5"
GOAL = "#16a34a"
NEXT = "#ea580c"
RUN = "#111827"
BAD = "#dc2626"
LIMIT = "#6b7280"
MAX_SLICES = 8


def _plt():
    try:
        import matplotlib

        if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
            matplotlib.use("Agg", force=False)
        from matplotlib import pyplot as plt
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError(
            "abbot.plots needs matplotlib: pip install matplotlib "
            '(or pip install "abbot[mcp]")'
        ) from None
    return plt


# -- what to show --------------------------------------------------------------------


def _goal(campaign: Campaign):
    """The policy's goal bound to the spec (outputs as positions), or None."""
    policy = campaign.policy
    if not hasattr(policy, "_goal_for"):
        return None
    return policy._goal_for(campaign.spec)


def _goal_outputs(goal, n_outputs: int) -> list[int]:
    """The outputs a goal reads, as positions."""
    if isinstance(goal, BandIntersection):
        return [j for j, b in enumerate(goal.bounds) if b is not None and _bounded(b)]
    if isinstance(goal, Wishlist):
        cols = {j for box in goal.bounds_list for j, b in enumerate(box)
                if b is not None and _bounded(b)}
        return sorted(cols)
    out = getattr(goal, "output", None)
    if isinstance(out, int):
        return [out]
    return list(range(n_outputs)) if n_outputs == 1 else []


def _bounded(band) -> bool:
    lo, hi = band
    return not (_inf(lo) and _inf(hi))


def _inf(v) -> bool:
    return v is None or (isinstance(v, float) and math.isinf(v))


def _pointwise(goal) -> bool:
    """Whether the goal judges each design on its own (so a slice can show it)."""
    return isinstance(goal, (BandIntersection, Wishlist, LevelSet))


def _output_columns(campaign: Campaign, output) -> list[int]:
    spec = campaign.spec
    if output is not None:
        names = [output] if isinstance(output, (str, int)) else list(output)
        cols = []
        for o in names:
            if isinstance(o, int):
                cols.append(o)
            elif o in spec.output_names:
                cols.append(spec.output_names.index(o))
            else:
                raise ValueError(f"unknown output {o!r}; the outputs are {spec.output_names}")
        return cols
    goal = _goal(campaign)
    cols = _goal_outputs(goal, spec.n_outputs) if goal is not None else []
    if not cols:
        limited = [j for j, o in enumerate(spec.outputs) if o.is_constrained]
        cols = sorted({*spec.objective_indices(), *limited})
    return cols or list(range(spec.n_outputs))


def _variable_columns(campaign: Campaign, variables) -> list[int]:
    spec = campaign.spec
    names = spec.design_names
    if variables is None:
        return list(range(len(names)))
    if isinstance(variables, str):
        variables = [variables]
    unknown = [v for v in variables if v not in names]
    if unknown:
        raise ValueError(f"unknown variable(s) {unknown}; the variables are {names}")
    return [names.index(v) for v in variables]


def _anchor(campaign: Campaign, at: Mapping | None) -> torch.Tensor:
    """The design the slices pass through: ``at`` over the recommended design."""
    spec = campaign.spec
    records = campaign.history.records
    rec = campaign.recommend()
    clean = [r for r in records if r.usable]
    base = records[rec[0]].candidate if rec else clean[-1].candidate
    x = spec.at_target(base.double().reshape(-1).clone())
    if at:
        position = {n: j for j, n in enumerate(spec.variable_names)}
        for name, value in at.items():
            if name not in position:
                raise ValueError(f"at: unknown variable {name!r}")
            d = spec.inputs[position[name]]
            x[position[name]] = d.encode(value) if isinstance(d, EnumeratedDomain) else float(value)
    return x


def _axis_values(d, n: int) -> torch.Tensor:
    if isinstance(d, EnumeratedDomain):
        return torch.tensor(d.codes, dtype=torch.double)
    lo, hi = float(d.lower), float(d.upper)
    if getattr(d, "log", False) and lo > 0:
        xs = torch.logspace(math.log10(lo), math.log10(hi), n, dtype=torch.double)
    else:
        xs = torch.linspace(lo, hi, n, dtype=torch.double)
    if isinstance(d, IntegerDomain):
        xs = torch.unique(xs.round())
    return xs


def _runs(campaign: Campaign):
    """``(designs, outputs, kinds)`` of the runs with outputs, in history order,
    using the outputs the model sees; kinds are "run" or "bad"."""
    history = campaign.scored_history()
    rows = [(r.candidate.double().reshape(-1), r.outputs.double().reshape(-1),
             "run" if r.usable else "bad")
            for r in history.records if r.outputs is not None and not r.fault]
    if not rows:
        return torch.empty(0), torch.empty(0), []
    X, Y, kinds = zip(*rows)
    return torch.stack(X), torch.stack(Y), list(kinds)


def _next_designs(campaign: Campaign) -> list[torch.Tensor]:
    """The outstanding designs, as tensors."""
    return [item.x.double().reshape(-1) for item in campaign._inflight.values()]


def _in_goal(campaign: Campaign, mean: torch.Tensor, grid: torch.Tensor) -> list[int]:
    policy = campaign.policy
    if not hasattr(policy, "_identify"):
        return []
    try:
        return list(policy._identify(campaign.spec, mean, grid))
    except Exception:  # noqa: BLE001 - a goal that can't run on this grid: skip the shading
        return []


# -- the model ----------------------------------------------------------------------


def model(
    campaign: Campaign,
    output: str | Sequence[str] | None = None,
    variables: str | Sequence[str] | None = None,
    at: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> Figure:
    """What the model predicts, with the runs and the goal.

    ``output``: which output(s) to show (default: the ones the goal reads, else the
    objectives). ``variables``: one variable for slices, two for a contour (default:
    every variable; a contour when there are exactly two). ``at``: ``{variable:
    value}`` to hold the other variables at, over the recommended design.
    """
    spec = campaign.spec
    gp = campaign.current_model()
    if gp is None:
        raise ValueError("No model yet: it needs at least one measured run.")
    outs = _output_columns(campaign, output)
    vars_ = _variable_columns(campaign, variables)
    anchor = _anchor(campaign, at)
    two = len(vars_) == 2 and all(
        not isinstance(spec.domain[j], EnumeratedDomain) for j in vars_
    )
    if two and (variables is not None or len(spec.domain) == 2):
        fig = _contours(campaign, gp, outs, vars_, anchor)
    else:
        if len(vars_) > MAX_SLICES:
            vars_ = vars_[:MAX_SLICES]
        fig = _slices(campaign, gp, outs, vars_, anchor)
    n = len(campaign.history)
    head = title or f"Model after {n} run{'s' if n != 1 else ''}"
    note = getattr(fig, "abbot_note", None)
    fig.suptitle(f"{head}\n{note}" if note else head, fontsize=11)
    return fig


def _slices(campaign, gp, outs, vars_, anchor) -> Figure:
    plt = _plt()
    spec = campaign.spec
    d = len(spec.domain)
    goal = _goal(campaign)
    X, Y, kinds = _runs(campaign)
    nxt = _next_designs(campaign)
    rows, cols = len(outs), len(vars_)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols + 0.6, 3.3 * rows + 0.8),
                             squeeze=False, layout="constrained")
    for c, j in enumerate(vars_):
        dom = spec.domain[j]
        xs = _axis_values(dom, 240)
        grid = anchor.repeat(len(xs), 1)
        grid[:, j] = xs
        mean, std = analysis._posterior_raw(gp, grid, spec)
        shade = _in_goal(campaign, mean, grid) if (d == 1 or _pointwise(goal)) else []
        enum = isinstance(dom, EnumeratedDomain)
        for r, k in enumerate(outs):
            ax = axes[r][c]
            m, s = mean[:, k].numpy(), std[:, k].numpy()
            x = xs.numpy()
            if enum:
                ax.errorbar(x, m, yerr=2 * s, fmt="o", color=MEAN, ecolor=BAND, capsize=4,
                            label="model ± 2σ")
                for i in shade:
                    ax.plot(x[i], m[i], "o", ms=13, mfc="none", mec=GOAL, mew=2)
                ax.set_xticks(x, [str(v) for v in dom.values])
            else:
                ax.fill_between(x, m - 2 * s, m + 2 * s, color=BAND, alpha=0.15,
                                lw=0, label="model ± 2σ")
                ax.plot(x, m, color=MEAN, lw=2, label="model mean")
                for lo, hi in _stretches(x, shade):
                    ax.axvspan(lo, hi, color=GOAL, alpha=0.10, lw=0)
                if getattr(dom, "log", False):
                    ax.set_xscale("log")
            if len(X):
                alpha = 1.0 if d == 1 else 0.55
                runs = [i for i, kind in enumerate(kinds) if kind == "run"]
                bad = [i for i, kind in enumerate(kinds) if kind == "bad"]
                ax.plot(X[runs, j].numpy(), Y[runs, k].numpy(), "o", color=RUN, ms=5,
                        alpha=alpha, label="runs" if d == 1 else "runs (other settings vary)")
                if bad:
                    ax.plot(X[bad, j].numpy(), Y[bad, k].numpy(), "x", color=BAD, ms=7,
                            label="bad-region runs")
                if kinds[-1] == "run":
                    ax.plot(X[-1, j].item(), Y[-1, k].item(), "o", ms=11, mfc="none",
                            mec=RUN, mew=1.5, label="latest run")
            for i, xn in enumerate(nxt):
                ax.axvline(xn[j].item(), color=NEXT, ls="--", lw=1.5,
                           label="next design" if i == 0 else None)
            if d > 1:
                ax.axvline(anchor[j].item(), color=LIMIT, ls=":", lw=1,
                           label="slice through")
            # After the data, so the band doesn't decide the axis range.
            _mark_goal(ax, goal, spec, k)
            _mark_limits(ax, spec.outputs[k])
            ax.set_xlabel(dom.name)
            ax.set_ylabel(spec.outputs[k].name)
            ax.grid(alpha=0.25)
    handles, labels = _legend_entries(fig)
    if goal is not None and (d == 1 or _pointwise(goal)):
        from matplotlib.patches import Patch

        handles.append(Patch(color=GOAL, alpha=0.25))
        labels.append("model's goal region")
    axes[0][0].legend(handles, labels, fontsize=8, loc="best")
    if d > 1:
        fig.abbot_note = "slices through " + ", ".join(
            f"{n}={_fmt(v)}" for n, v in spec.as_dict(anchor).items()
        )
    return fig


def _contours(campaign, gp, outs, vars_, anchor) -> Figure:
    plt = _plt()
    spec = campaign.spec
    goal = _goal(campaign)
    a, b = vars_
    da, db = spec.domain[a], spec.domain[b]
    xa, xb = _axis_values(da, 60), _axis_values(db, 60)
    ga, gb = torch.meshgrid(xa, xb, indexing="xy")
    grid = anchor.repeat(ga.numel(), 1)
    grid[:, a], grid[:, b] = ga.reshape(-1), gb.reshape(-1)
    mean, std = analysis._posterior_raw(gp, grid, spec)
    full = len(spec.domain) == 2
    region = _in_goal(campaign, mean, grid) if (full or _pointwise(goal)) else []
    mask = torch.zeros(ga.numel())
    mask[region] = 1.0
    mask = mask.reshape(ga.shape).numpy()
    X, _, kinds = _runs(campaign)
    nxt = _next_designs(campaign)
    fig, axes = plt.subplots(len(outs), 2, figsize=(10.5, 4.2 * len(outs) + 0.6),
                             squeeze=False, layout="constrained")
    for r, k in enumerate(outs):
        for c, (values, label, cmap) in enumerate((
            (mean[:, k], f"{spec.outputs[k].name} (model mean)", "viridis"),
            (2 * std[:, k], "uncertainty (2σ)", "magma"),
        )):
            ax = axes[r][c]
            z = values.reshape(ga.shape).numpy()
            cs = ax.contourf(xa.numpy(), xb.numpy(), z, levels=20, cmap=cmap)
            fig.colorbar(cs, ax=ax, label=label)
            if region:
                ax.contour(xa.numpy(), xb.numpy(), mask, levels=[0.5], colors="white",
                           linewidths=2, linestyles="--")
            out = spec.outputs[k]
            limits = [v for v in (out.min_val, out.max_val) if v is not None]
            m_k = mean[:, k].reshape(ga.shape).numpy()
            if limits and m_k.min() < max(limits) and m_k.max() > min(limits):
                ax.contour(xa.numpy(), xb.numpy(), m_k, levels=sorted(limits),
                           colors="#f9fafb", linewidths=1.5, linestyles=":")
            if len(X):
                runs = [i for i, kind in enumerate(kinds) if kind == "run"]
                bad = [i for i, kind in enumerate(kinds) if kind == "bad"]
                ax.plot(X[runs, a].numpy(), X[runs, b].numpy(), "o", mfc="white", mec=RUN,
                        ms=6, label="runs")
                if bad:
                    ax.plot(X[bad, a].numpy(), X[bad, b].numpy(), "x", color=BAD, ms=7,
                            label="bad-region runs")
            for i, xn in enumerate(nxt):
                ax.plot(xn[a].item(), xn[b].item(), "*", color=NEXT, ms=15, mec="white",
                        label="next design" if i == 0 else None)
            if getattr(da, "log", False):
                ax.set_xscale("log")
            if getattr(db, "log", False):
                ax.set_yscale("log")
            ax.set_xlabel(da.name)
            ax.set_ylabel(db.name)
    handles, labels = axes[0][0].get_legend_handles_labels()
    from matplotlib.lines import Line2D

    if region:
        handles.append(Line2D([], [], color="white", ls="--", lw=2))
        labels.append("model's goal region")
    if any(spec.outputs[k].is_constrained for k in outs):
        handles.append(Line2D([], [], color="#f9fafb", ls=":", lw=1.5))
        labels.append("output limit (model mean)")
    leg = axes[0][0].legend(handles, labels, fontsize=8, loc="best", facecolor="#e5e7eb")
    leg.get_frame().set_alpha(0.9)
    if not full:
        fig.abbot_note = "slice through " + ", ".join(
            f"{n}={_fmt(v)}" for n, v in spec.as_dict(anchor).items()
            if n not in (da.name, db.name)
        )
    return fig


def _legend_entries(fig) -> tuple[list, list]:
    """Every panel's legend entries, each label once."""
    handles, labels = [], []
    for ax in fig.axes:
        for h, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(h)
                labels.append(label)
    return handles, labels


def _mark_goal(ax, goal, spec, k: int) -> None:
    """The goal on output ``k``'s axis: a band, or a threshold."""
    if isinstance(goal, BandIntersection):
        boxes = [goal.bounds]
    elif isinstance(goal, Wishlist):
        boxes = goal.bounds_list
    elif isinstance(goal, LevelSet) and goal.output == k:
        side = "≤" if goal.upper else "≥"
        ax.axhline(goal.threshold, color=GOAL, lw=2, label=f"goal: {side} {_fmt(goal.threshold)}")
        return
    else:
        return
    for i, box in enumerate(boxes):
        band = box[k] if k < len(box) else None
        if band is None or not _bounded(band):
            continue
        lo, hi = band
        lo = -math.inf if _inf(lo) else lo
        hi = math.inf if _inf(hi) else hi
        ymin, ymax = ax.get_ylim()
        ax.axhspan(max(lo, -1e300), min(hi, 1e300), color=GOAL, alpha=0.15, lw=0,
                   label="goal band" if i == 0 else None)
        for edge in (lo, hi):
            if math.isfinite(edge):
                ax.axhline(edge, color=GOAL, lw=1)
        ax.set_ylim(ymin, ymax)


def _mark_limits(ax, out) -> None:
    for v, side in ((out.min_val, "min"), (out.max_val, "max")):
        if v is not None:
            ax.axhline(v, color=LIMIT, ls="--", lw=1.2, label=f"{side} {out.name} {_fmt(v)}")


def _stretches(x, idx: list[int]) -> list[tuple[float, float]]:
    """Contiguous runs of grid indices as (lo, hi) spans, padded half a step."""
    if not idx:
        return []
    idx = sorted(idx)
    spans, start, prev = [], idx[0], idx[0]
    for i in idx[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        lo = x[start] - (x[start] - x[start - 1]) / 2 if start > 0 else x[start]
        hi = x[prev] + (x[prev + 1] - x[prev]) / 2 if prev + 1 < len(x) else x[prev]
        spans.append((float(lo), float(hi)))
        if i is not None:
            start = prev = i
    return spans


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    return f"{v:.4g}"


# -- progress and check ---------------------------------------------------------


def progress(campaign: Campaign, title: str | None = None) -> Figure:
    """How the campaign is going: the best so far (one objective), the dominated
    hypervolume (several), or each run's goal outputs against the goal (a goal
    without objectives)."""
    plt = _plt()
    spec = campaign.spec
    history = campaign.scored_history()
    steps = list(range(1, len(history) + 1))
    if spec.n_objectives >= 1:
        fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
        if spec.n_objectives == 1:
            k = spec.objective_indices()[0]
            best = analysis.convergence(history, spec)
            each = [float(r.outputs.reshape(-1)[k]) if r.outputs is not None and r.usable
                    else math.nan for r in history.records]
            ax.plot(steps, each, "o", color=RUN, ms=5, alpha=0.6, label="each run")
            ax.step(steps, best, where="post", color=MEAN, lw=2, label="best so far")
            ax.set_ylabel(spec.outputs[k].name)
        else:
            ax.plot(steps, analysis.hypervolume(history, spec), color=MEAN, lw=2,
                    marker="o", ms=4, label="dominated hypervolume")
            ax.set_ylabel("hypervolume")
        ax.set_xlabel("run")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    else:
        goal = _goal(campaign)
        cols = _goal_outputs(goal, spec.n_outputs) if goal is not None else []
        cols = cols or list(range(spec.n_outputs))
        fig, axes = plt.subplots(len(cols), 1, figsize=(6.5, 2.9 * len(cols) + 0.6),
                                 squeeze=False, layout="constrained")
        for ax, k in zip(axes[:, 0], cols):
            each = [float(r.outputs.reshape(-1)[k]) if r.outputs is not None and r.usable
                    else math.nan for r in history.records]
            ax.plot(steps, each, "o-", color=RUN, ms=5, lw=0.8, alpha=0.8, label="each run")
            _mark_goal(ax, goal, spec, k)
            ax.set_ylabel(spec.outputs[k].name)
            ax.grid(alpha=0.25)
        axes[-1, 0].set_xlabel("run")
        axes[0, 0].legend(fontsize=8)
    n = len(history)
    fig.suptitle(title or f"Progress after {n} run{'s' if n != 1 else ''}", fontsize=12)
    return fig


def check(campaign: Campaign, title: str | None = None) -> Figure:
    """Each run's leave-one-out error: how far it lies from what the other runs
    predict, in standard deviations, per output. Points beyond ±3σ are what
    :meth:`Campaign.check` calls out; a spread much wider than ±2σ means the model is
    overconfident."""
    plt = _plt()
    from abbot.diagnostics import OUTLIER_Z

    report = campaign.check()
    fits = [f for f in report.fits if f.z]
    if not fits:
        raise ValueError(
            "No leave-one-out errors yet: they need a Gaussian process and at least "
            "5 clean runs."
        )
    fig, axes = plt.subplots(len(fits), 1, figsize=(6.5, 2.9 * len(fits) + 0.6),
                             squeeze=False, layout="constrained")
    for ax, fit in zip(axes[:, 0], fits):
        runs = sorted(fit.z)
        z = [fit.z[s] for s in runs]
        x = [s + 1 for s in runs]
        ax.axhspan(-2, 2, color=BAND, alpha=0.10, lw=0, label="±2σ")
        for edge in (-OUTLIER_Z, OUTLIER_Z):
            ax.axhline(edge, color=BAD, ls="--", lw=1)
        far = [abs(v) > OUTLIER_Z for v in z]
        ax.plot(x, z, "o", color=RUN, ms=5, label="run")
        ax.plot([xi for xi, f in zip(x, far) if f], [v for v, f in zip(z, far) if f], "o",
                color=BAD, ms=8, label=f"beyond ±{OUTLIER_Z:g}σ")
        r2 = f", leave-one-out R² {fit.r2:.2f}" if fit.r2 is not None else ""
        ax.set_title(f"{fit.name}{r2}", fontsize=10)
        ax.set_ylabel("error (σ)")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("run")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title or "Model check: each run against the others", fontsize=12)
    return fig


__all__ = ["check", "model", "progress"]
