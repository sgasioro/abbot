"""Experiment-side protocols: how the world is probed and supervised.

These are the seams the lab-experiment orchestration layer plugs into — one of the two
genuinely-novel surfaces abbot owns on top of botorch (the other is the
composite objective + aux layer, :mod:`abbot.composite`).

``Experiment`` is deliberately a plain callable contract: a GUI action, an HTTP
reactor call, a simulator, or (later) an agent driving an instrument all satisfy
it. This is the headless-first property — the loop never knows which it is.
"""

from __future__ import annotations

import json
import os
import shlex
import string
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

# A design as the loop hands it out: {variable name: value}, in spec.variable_names
# order. The same shape ``Campaign.suggest`` returns.
Design = Mapping[str, float]


@runtime_checkable
class Experiment(Protocol):
    """Run one design and return what was measured. May block (slow hardware).

    ``design`` is ``{variable name: value}`` in ``spec.variable_names`` order — the
    same dict ``Campaign.suggest`` returns. Return either the outputs directly (a
    tensor / sequence / float in ``spec.outputs`` order, or a ``{output name:
    value}`` dict) or a raw observation (a spectrum, an image, ...) that the
    ``evaluate`` callable reduces to outputs.

    An experiment written against vectors (a vectorized simulator) can be
    wrapped in :class:`TensorExperiment`.

    Hardware that runs several designs at once implements
    :class:`BatchExperiment` instead.

    Optionally, ``cost(design) -> float | None`` reports what a run cost, after it
    ran (core-hours from the scheduler, money). ``Campaign.step`` records that
    instead of the wall-clock time it measured, which would include queueing;
    ``None`` keeps the measured time. It is asked about failed runs too.
    """

    def run(self, design: Design) -> Any: ...


@runtime_checkable
class BatchExperiment(Protocol):
    """Run several designs at once (a plate, several reactors).

    ``designs`` is a list of design dicts (as for :class:`Experiment`); return one
    observation per design, in order, or an exception object for a design that
    failed (raising fails them all). The loop sends every step through
    ``run_batch``, batches of one included, so there is one code path whatever
    the batch size. An object defining both ``run`` and ``run_batch`` is treated
    as a ``BatchExperiment``. It may define ``cost(design)`` as an
    :class:`Experiment` may.
    """

    def run_batch(self, designs: list[Design]) -> list[Any]: ...


class TensorExperiment:
    """Adapt a vector-input experiment (``run(x: Tensor)``) to the design-dict
    contract. ``x`` is the ``(d,)`` tensor in ``spec.variable_names`` order, in
    ``dtype`` (double by default). A wrapped ``run_batch(X)`` gets the ``(q, d)``
    tensor instead, and the wrapper is then a :class:`BatchExperiment`. A wrapped
    ``cost(x)`` gets the ``(d,)`` tensor.

    A spec with labeled values (``EnumeratedDomain(values=["water", ...])``) needs
    ``spec=``: the tensor then holds each label's code, its position in ``values``.
    """

    def __init__(
        self, experiment: Any, dtype: torch.dtype = torch.double, *, spec: Any = None
    ) -> None:
        self.experiment = experiment
        self.dtype = dtype
        self.spec = spec
        if hasattr(experiment, "run_batch"):
            self.run_batch = self._run_batch
        else:
            self.run = self._run
        if hasattr(experiment, "cost"):
            self.cost = self._cost

    def _tensor(self, design: Design) -> torch.Tensor:
        if self.spec is not None:
            return self.spec.as_tensor(design).to(self.dtype)
        if any(isinstance(v, str) for v in design.values()):
            raise TypeError(
                f"design {dict(design)} has labels; pass TensorExperiment(..., spec=spec) "
                "so they become codes"
            )
        return torch.tensor(list(design.values()), dtype=self.dtype)

    def _cost(self, design: Design) -> Any:
        return self.experiment.cost(self._tensor(design))

    def _run(self, design: Design) -> Any:
        return self.experiment.run(self._tensor(design))

    def _run_batch(self, designs: list[Design]) -> Any:
        X = torch.stack([self._tensor(d) for d in designs])
        return self.experiment.run_batch(X)


@dataclass
class Verdict:
    """What a ``Supervisor`` decided about the run so far."""

    retry: bool = False  # this observation is unusable; re-run the same candidate
    halt: str | None = None  # apparatus is broken; human-readable reason
    stop: str | None = None  # done (budget spent, converged); not a fault. Reason.
    # The design itself is the cause: no usable result here, and none on a re-run.
    # A reason for people, or True. Recorded as a bad-region run; no retry.
    bad_region: str | bool | None = None


BAD_REGION = "bad region"  # the reason stored for Verdict(bad_region=True)


def _bad_region_reason(bad: str | bool | None) -> str | None:
    """A verdict's ``bad_region`` as the reason to store, or None."""
    if bad is None or bad is False:
        return None
    return BAD_REGION if bad is True else str(bad)


class ManualExperiment:
    """Human-in-the-loop Experiment: a person measures the proposed candidate.

    This is the lab tool's "experiment" — the loop proposes a design, a human
    runs the real-world reaction/synthesis and types back the measured value(s).
    By default it prompts on stdin, but a ``responder`` callable can be injected
    (so the same flow is headless-testable, and so a web frontend or an agent can
    supply the measurement instead of a terminal). The responder receives the
    design dict and returns the measured output(s).
    ``variable_names`` is accepted for backward compatibility and ignored (the
    design dict carries the names).
    """

    def __init__(
        self,
        responder: Any | None = None,
        variable_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ) -> None:
        self.responder = responder
        self.variable_names = variable_names
        self.output_names = output_names or ["value"]

    def run(self, design: Design) -> Any:
        if self.responder is not None:
            out = self.responder(design)
            if isinstance(out, (torch.Tensor, Mapping)):
                return out
            return torch.as_tensor(out, dtype=torch.double)
        return self._prompt(design)

    def _prompt(self, design: Design) -> torch.Tensor:  # pragma: no cover
        setting = ", ".join(f"{n}={v:.4g}" for n, v in design.items())
        print(f"\nRun the experiment at: {setting}")
        measured = []
        for name in self.output_names:
            raw = input(f"  measured {name}: ")
            measured.append(float(raw))
        return torch.tensor(measured, dtype=torch.double)


class CommandExperiment:
    """Run a command for each design; it prints what it measured as JSON.

    ``command`` names the design's values in braces, as ``str.format`` does::

        CommandExperiment("./measure.sh {temp} {conc}")
        CommandExperiment("python rig.py --temp {temp:.1f} --solvent {solvent}")

    Each brace is filled in with the value (a format spec like ``:.1f`` is honored),
    one argument per word of ``command``, so a value is never split or read by a
    shell. Write a literal brace as ``{{`` or ``}}``. ``shell=True`` runs the command through the shell instead (pipes,
    ``&&``), each value quoted. The design is also written to the command's stdin
    as JSON (``{"temp": 54.2, "conc": 0.1}``), for a script that would rather read
    that.

    The **last line** the command prints is read as JSON (lines before it are
    free for logging): a ``{output name: value}`` object, a list in
    ``spec.outputs`` order, or a number for a single output. In an object,
    ``"cost"`` (unless an output has that name) is recorded as what the run cost;
    with ``spec``, keys that aren't outputs are kept with the run instead of
    refused. A command that exits non-zero, prints no JSON or runs past
    ``timeout`` seconds fails the run, with the reason and the end of its stderr
    recorded; the campaign retries it, as it would any failed run.

    ``spec`` is the spec the command works with: its names are checked against
    the braces up front, and the UI holds a setup to it (as for any experiment
    with a ``spec``). ``cwd`` and ``env`` (added to the current environment) are
    passed to the command.
    """

    def __init__(
        self,
        command: str,
        *,
        spec: Any = None,
        timeout: float | None = None,
        cwd: str | os.PathLike | None = None,
        env: Mapping[str, str] | None = None,
        shell: bool = False,
    ) -> None:
        self.command = command
        self.spec = spec
        self.timeout = timeout
        self.cwd = cwd
        self.env = None if env is None else {**os.environ, **{k: str(v) for k, v in env.items()}}
        self.shell = shell
        self.words = None if shell else shlex.split(command)
        if not shell and not self.words:
            raise ValueError("CommandExperiment: the command is empty.")
        # The variables it reads, in the order the command names them, and those
        # it formats as whole numbers ({n:d}).
        parsed = [(field.split(".")[0].split("[")[0], fmt or "")
                  for _, field, fmt, _ in string.Formatter().parse(command) if field]
        self.fields = list(dict.fromkeys(name for name, _ in parsed))
        self.whole = list(dict.fromkeys(name for name, fmt in parsed if fmt.endswith("d")))
        if spec is not None:
            unknown = [f for f in self.fields if f not in spec.variable_names]
            if unknown:
                raise ValueError(
                    f"{self.name}: {{{unknown[0]}}} isn't a variable; the variables are "
                    f"{', '.join(spec.variable_names)}."
                )
        self._costs: dict[tuple, float] = {}

    @property
    def name(self) -> str:
        """The program, for messages and the UI: ``measure.sh``."""
        first = shlex.split(self.command)[0] if self.command.strip() else "command"
        return os.path.basename(first)

    def __repr__(self) -> str:
        return f"CommandExperiment({self.command!r})"

    def args(self, design: Design) -> list[str] | str:
        """The command for ``design``: a list of arguments, or (``shell=True``) a
        string with each value quoted. A whole number held as a float (``3.0``)
        fills a ``{n:d}`` as ``3``."""
        values = dict(design)
        for name in self.whole:
            value = _plain(values.get(name))
            if isinstance(value, float) and value.is_integer():
                values[name] = int(value)
        try:
            if self.shell:
                quoted = {k: _Quoted(v) for k, v in values.items()}
                return self.command.format_map(quoted)
            return [word.format_map(values) for word in self.words]
        except KeyError as exc:
            raise KeyError(
                f"{self.name}: {{{exc.args[0]}}} isn't a variable; the design has "
                f"{', '.join(values)}"
            ) from None

    def run(self, design: Design) -> Any:
        args = self.args(design)
        payload = json.dumps({k: _plain(v) for k, v in dict(design).items()})
        try:
            done = subprocess.run(
                args, input=payload, capture_output=True, text=True, shell=self.shell,
                timeout=self.timeout, cwd=self.cwd, env=self.env, check=False,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{self.name} ran past {self.timeout:g} s") from None
        except FileNotFoundError:
            raise RuntimeError(
                f"{self.name}: no such program (from {self.command!r}"
                + (f", in {self.cwd}" if self.cwd else "") + ")"
            ) from None
        except PermissionError:
            raise RuntimeError(f"{self.name} isn't executable (chmod +x?)") from None
        if done.returncode != 0:
            raise RuntimeError(
                f"{self.name} exited with code {done.returncode}" + _tail(done.stderr))
        lines = [line for line in done.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"{self.name} printed nothing" + _tail(done.stderr))
        try:
            result = json.loads(lines[-1])
        except json.JSONDecodeError:
            raise RuntimeError(
                f"{self.name}'s last line isn't JSON: {lines[-1][:200]!r}") from None
        return self._split(result, design)

    def _split(self, result: Any, design: Design) -> Any:
        """Outputs, with ``"cost"`` set aside and (given ``spec``) other keys kept as
        the run's extras: ``(outputs, extras)``."""
        if not isinstance(result, dict):
            return result
        outputs = set(self.spec.output_names) if self.spec is not None else None
        if "cost" in result and (outputs is None or "cost" not in outputs):
            cost = result.pop("cost")
            if cost is not None:
                self._costs[_key(design)] = float(cost)
        if outputs is None:
            return result
        extras = {k: v for k, v in result.items() if k not in outputs}
        values = {k: v for k, v in result.items() if k in outputs}
        return (values, extras) if extras else values

    def cost(self, design: Design) -> float | None:
        """What the command said the run cost (``"cost"`` in its JSON), if it did."""
        return self._costs.pop(_key(design), None)


class _Quoted:
    """A value for ``shell=True``: formatted as asked, then shell-quoted."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __format__(self, spec: str) -> str:
        return shlex.quote(format(self.value, spec))


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, torch.Tensor) else value


def _key(design: Design) -> tuple:
    return tuple(sorted((k, str(_plain(v))) for k, v in dict(design).items()))


def _tail(stderr: str, lines: int = 10) -> str:
    """The end of a command's stderr, to say why it failed."""
    text = "\n".join((stderr or "").strip().splitlines()[-lines:])
    return f": {text[-1500:]}" if text else ""


class MockAnalyticExperiment:
    """A mock experiment whose answers come from a closed-form formula, for
    offline runs and demos. (For answers that already exist, a measured or
    simulated grid, see :class:`MockDatasetExperiment`.)

    Produces smooth, deterministic (seeded) RAW-physical outputs for WHATEVER
    ``ProblemSpec`` you built — any number of variables, any number of outputs
    (one column per ``spec.outputs``), optional observation noise. This is what
    makes the UI self-demonstrating: with
    a programmatic ``Experiment`` attached, auto-run (run/pause/step/stop) becomes
    available and a whole campaign drives itself without typed measurements.

    Each objective is a smooth Gaussian bump over the design space with its own
    (seeded) optimum location, so a single-objective problem has a clear maximum
    and a multi-objective problem has a genuine trade-off (the bumps peak in
    different places). Outputs are RAW and always "larger is better" in shape; the
    engine applies ``objective_signs`` for MINIMIZE objectives, so the mock does
    NOT need to know directions. Values are mapped to a friendly ~[0, 100] range.

    Deterministic: the landscape is fixed by ``seed`` and, when ``noise`` > 0, the
    per-evaluation noise is seeded by the candidate itself (so re-measuring the
    same point gives the same value — honest for a resumable run).

    **With sources** the bumps live on the design variables, and a run away from a
    source's accurate end is biased: each output gets ``source_bias`` times a
    second, differently placed bump (scaled to the same range), fading to nothing
    at the accurate end (linearly on the source's model scale: log for
    ``log=True``, flipped for ``accurate="low"``). Each level of an unordered source
    other than the target gets its own bias size, between half and all of
    ``source_bias``. ``cost(design)`` then reports a simulated cost that
    ``Campaign.step`` records: ``cost_ratio ** u``, with ``u`` the mean over
    sources of how close each is to its accurate end (1 for an unordered source at
    its target, else 0), so a run at the accurate end costs ``cost_ratio`` times the
    cheapest. Without sources ``cost`` returns ``None`` and the measured time is
    kept.
    """

    def __init__(
        self,
        spec: Any,
        *,
        seed: int = 0,
        noise: float = 0.0,
        source_bias: float = 0.3,
        cost_ratio: float = 100.0,
    ) -> None:
        self.spec = spec
        self.seed = seed
        self.noise = float(noise)
        self.source_bias = float(source_bias)
        self.cost_ratio = float(cost_ratio)
        bounds = spec.bounds_tensor()
        self._lo = bounds[0]
        self._hi = bounds[1]
        n_design = len(spec.design_names)
        m = spec.n_outputs
        # Per-objective optima and widths, fixed by the seed.
        g = torch.Generator().manual_seed(seed)
        self._centers = torch.rand(m, n_design, generator=g, dtype=torch.double)
        self._widths = 0.15 + 0.25 * torch.rand(m, n_design, generator=g, dtype=torch.double)
        # The bias cheap runs add: another bump per output, and per-level sizes.
        self._bias_centers = torch.rand(m, n_design, generator=g, dtype=torch.double)
        self._bias_widths = 0.15 + 0.25 * torch.rand(m, n_design, generator=g, dtype=torch.double)
        self._level_bias = [
            {v: 0.5 + 0.5 * float(torch.rand(1, generator=g, dtype=torch.double))
             for v in s.domain.codes}
            if not s.is_ordered else None
            for s in spec.sources or []
        ]

    def _bump(self, xn, centers, widths):
        sq = ((xn.unsqueeze(0) - centers) / widths) ** 2  # (m, n_design)
        return 100.0 * torch.exp(-0.5 * sq.sum(dim=-1))  # (m,)

    def _accuracy(self, design: Design) -> list[float]:
        """Per source: 1 at the accurate end (an unordered source's target), 0 at
        the other end."""
        import math

        out = []
        x = self.spec.as_tensor(design).tolist()
        for i, source in zip(self.spec.source_indices(), self.spec.sources or []):
            v = x[i]
            if not source.is_ordered:
                out.append(1.0 if v == source.target_code else 0.0)
                continue
            from abbot.problem import EnumeratedDomain

            d = source.domain
            if isinstance(d, EnumeratedDomain):
                lo, hi = min(d.codes), max(d.codes)
            else:
                lo, hi = d.lower, d.upper
            if source.is_log:
                v, lo, hi = math.log(v), math.log(lo), math.log(hi)
            u = (v - lo) / (hi - lo) if hi > lo else 1.0
            out.append(min(max(u if source.accurate == "high" else 1.0 - u, 0.0), 1.0))
        return out

    def run(self, design: Design) -> torch.Tensor:
        x = self.spec.as_tensor(design)
        span = (self._hi - self._lo).clamp(min=1e-9)
        xn = ((x - self._lo) / span).clamp(0.0, 1.0)  # normalized to [0, 1]^d
        n_design = len(self.spec.design_names)
        vals = self._bump(xn[:n_design], self._centers, self._widths)  # (m,) in (0, 100]
        if self.spec.sources:
            bias = 0.0
            for source, u, levels in zip(
                self.spec.sources, self._accuracy(design), self._level_bias, strict=True
            ):
                code = source.domain.encode(design[source.name]) if levels else None
                size = (1.0 - u) if levels is None else levels[code] * (1.0 - u)
                bias += self.source_bias * size
            vals = vals + bias * self._bump(xn[:n_design], self._bias_centers, self._bias_widths)
        if self.noise > 0.0:
            # Seed noise by the candidate so a re-measurement repeats exactly.
            key = int(abs(hash((self.seed, tuple(round(float(v), 6) for v in xn.tolist())))) % (2**31))
            gn = torch.Generator().manual_seed(key)
            vals = vals + self.noise * torch.randn(vals.shape, generator=gn, dtype=torch.double)
        return vals

    def cost(self, design: Design) -> float | None:
        """The simulated cost of a run (``None`` without sources: measured time)."""
        if not self.spec.sources:
            return None
        u = self._accuracy(design)
        return self.cost_ratio ** (sum(u) / len(u))


class MockDatasetExperiment:
    """A mock experiment over a fixed set of designs whose outputs are already
    known: a measured dataset or a simulated grid, replayed as if run.

    ``X`` ``(N, d)`` holds the designs (columns in ``spec.variable_names`` order)
    and ``Y`` ``(N, m)`` their raw outputs (columns in ``spec.outputs`` order).
    ``run(design)`` returns that design's row of ``Y``; a design not in the set
    raises ``ValueError``, so pair it with a policy that proposes only from
    :attr:`candidates` (``BAXPolicy(candidates=...)``,
    ``DiscreteArgmax(choices=...)``). A design listed more than once (repeat
    measurements) returns its first row.
    """

    def __init__(self, spec: Any, X: torch.Tensor, Y: torch.Tensor) -> None:
        X = torch.as_tensor(X, dtype=torch.double)
        Y = torch.as_tensor(Y, dtype=torch.double)
        Y = Y.reshape(len(Y), -1)
        if X.ndim != 2 or X.shape[1] != len(spec.variable_names):
            raise ValueError(
                f"X must be (N, {len(spec.variable_names)}) for variables "
                f"{spec.variable_names}, got {tuple(X.shape)}"
            )
        if Y.shape != (X.shape[0], spec.n_outputs):
            raise ValueError(
                f"Y must be ({X.shape[0]}, {spec.n_outputs}) for outputs "
                f"{spec.output_names}, got {tuple(Y.shape)}"
            )
        self.spec = spec
        self._X = X
        self._Y = Y

    @classmethod
    def from_history(cls, history: Any, spec: Any) -> MockDatasetExperiment:
        """Replay the clean, measured records of ``history`` (e.g. a dataset loaded
        with :func:`abbot.io.history_from_csv`)."""
        records = [r for r in history.clean.records if r.outputs is not None]
        if not records:
            raise ValueError("history has no clean records with outputs")
        X = torch.stack([r.candidate.double().reshape(-1) for r in records])
        Y = torch.stack([r.outputs.double().reshape(-1) for r in records])
        return cls(spec, X, Y)

    @property
    def candidates(self) -> torch.Tensor:
        """The ``(N, d)`` designs this experiment can run."""
        return self._X

    def run(self, design: Design) -> torch.Tensor:
        x = self.spec.as_tensor(design)
        hits = torch.isclose(self._X, x.unsqueeze(0)).all(dim=-1).nonzero()
        if len(hits) == 0:
            raise ValueError(f"design {dict(design)} is not in this dataset")
        return self._Y[int(hits[0, 0])].clone()


def __getattr__(name: str):
    # ``Supervisor`` moved to abbot.supervision (a class now, not a protocol); this
    # keeps ``from abbot.experiment import Supervisor`` working.
    if name == "Supervisor":
        from abbot.supervision import Supervisor

        return Supervisor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
