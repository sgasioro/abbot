"""Your own experiment in the UI: ``--experiment module:Name`` or :func:`abbot.ui.serve`.

The UI runs each design through it, as it runs the built-in simulation. ``Name``
is an experiment (an object with ``run(design)`` or ``run_batch(designs)``,
:class:`abbot.experiment.Experiment`), or a class or function that makes one. A
maker is called when a campaign starts, with the campaign's spec if it takes an
argument, so one experiment class can serve any setup.

An experiment can declare the spec it works with (a ``spec`` attribute, or
``--spec``): which knobs it reads, how far each can go, and what it returns. The
setup then starts from it and stays within it: the same knobs and measurements,
ranges and choices only narrowed, its travel limits and design constraints kept.
The goal is free. Without a declared spec any setup is allowed.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from abbot.problem import (
    ContinuousDomain,
    EnumeratedDomain,
    IntegerDomain,
    ProblemSpec,
)


def _is_experiment(obj) -> bool:
    return not isinstance(obj, type) and (hasattr(obj, "run") or hasattr(obj, "run_batch"))


class YourExperiment:
    """An experiment for the UI to run: ``make(spec)`` gives the object whose
    ``run`` / ``run_batch`` measures a design; ``evaluate`` (optional) reduces what
    it returns to the outputs. ``name`` labels it on the page, ``source`` says where
    it came from (``rig:Rig``).

    ``spec`` is the spec it works with (a :class:`ProblemSpec`, its JSON, or a
    campaign.json); by default the target's own ``spec`` attribute, if it has one.
    A campaign's spec must then stay within it (see :func:`within`)."""

    def __init__(self, target, *, spec=None, evaluate: Callable | None = None,
                 name: str | None = None, source: str | None = None,
                 reads: list[str] | None = None) -> None:
        if not _is_experiment(target) and not callable(target):
            raise TypeError(
                f"{source or target!r} is neither an experiment (an object with run(design) "
                "or run_batch(designs)) nor a class or function that makes one."
            )
        self.target = target
        self.evaluate = evaluate
        self.name = name or _name(target)
        self.source = source
        declared = spec if spec is not None else getattr(target, "spec", None)
        if spec is None and not isinstance(declared, ProblemSpec):
            declared = None  # an attribute named spec that isn't one: not a declaration
        self.spec: ProblemSpec | None = None if declared is None else as_spec(declared)
        # The variables it's known to read (a command's {names}), when it says.
        self.reads = list(reads or [])

    def problem(self, spec: ProblemSpec) -> str | None:
        """Why a campaign on ``spec`` can't run on this experiment, or None."""
        missing = [n for n in self.reads if n not in spec.variable_names]
        if missing:
            return (f"{self.name} reads {listed(missing)}, which "
                    f"{'is' if len(missing) == 1 else 'are'}n't a knob here; the knobs are "
                    f"{listed(spec.variable_names)}.")
        return None if self.spec is None else within(self.spec, spec, self.name)

    def make(self, spec: ProblemSpec):
        """The experiment for a campaign on ``spec``. Raises if ``spec`` goes beyond
        the one the experiment declares."""
        problem = self.problem(spec)
        if problem:
            raise ValueError(problem)
        if _is_experiment(self.target):
            return self.target
        try:
            takes_spec = bool(inspect.signature(self.target).parameters)
        except (TypeError, ValueError):  # a builtin without a signature: try plain
            takes_spec = False
        made = self.target(spec) if takes_spec else self.target()
        if not _is_experiment(made):
            raise TypeError(
                f"{self.source or self.name} made {type(made).__name__}, which has no "
                "run(design) or run_batch(designs)."
            )
        return made

    def outputs(self, spec: ProblemSpec) -> Callable:
        """``evaluate`` for the run controller: yours (or as is), with a returned
        dict's names checked against ``spec`` first, so a mismatch says so plainly."""
        evaluate = self.evaluate or (lambda o: o)

        def checked(observation):
            out = evaluate(observation)
            values = out[0] if isinstance(out, tuple) else out
            if isinstance(values, Mapping) and set(values) != set(spec.output_names):
                raise ValueError(
                    f"{self.name} returned {listed(values)}; this setup expects "
                    f"{listed(spec.output_names)}. Name the measurements as {self.name} "
                    "does, or change what it returns."
                )
            return out

        return checked


def as_spec(value) -> ProblemSpec:
    """A spec from a :class:`ProblemSpec`, its JSON (a dict), a campaign.json (its
    ``"spec"``), or a path to either."""
    if isinstance(value, ProblemSpec):
        return value
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).expanduser().read_text())
    if isinstance(value, Mapping) and "spec" in value and "domain" not in value:
        value = value["spec"]
    return ProblemSpec.model_validate(value)


def within(declared: ProblemSpec, spec: ProblemSpec, who: str = "The experiment") -> str | None:
    """Why ``spec`` goes beyond ``declared``, the spec an experiment works with, or
    None if it stays within: the same knobs (and sources) and measurements, each
    range or list of choices the same or narrower, every travel limit and design
    constraint kept or tightened. Directions, targets, output limits, the goal and
    the model are free."""
    for what, mine, theirs in (
        ("sets", declared.domain, spec.domain),
        ("has sources", [s.domain for s in declared.sources or []],
         [s.domain for s in spec.sources or []]),
    ):
        want, got = [d.name for d in mine], [d.name for d in theirs]
        if set(want) != set(got):
            return (f"{who} {what} {listed(want) or 'none'}; this setup has "
                    f"{listed(got) or 'none'}.")
        by_name = {d.name: d for d in mine}
        for d in theirs:
            problem = _domain_within(by_name[d.name], d, who)
            if problem:
                return problem
    want, got = declared.output_names, spec.output_names
    if set(want) != set(got):
        return f"{who} measures {listed(want)}; this setup has {listed(got)}."
    kept = spec.domain_constraints or []
    for c in declared.domain_constraints or []:
        if not any(k.coeffs == c.coeffs and _tighter(k.min_val, c.min_val, max)
                   and _tighter(k.max_val, c.max_val, min) for k in kept):
            return (f"{who} keeps {_constraint_words(c)}; this setup drops or loosens "
                    "it.")
    return None


def _domain_within(mine, theirs, who: str) -> str | None:
    name = mine.name
    if type(theirs) is not type(mine):
        return f"{name}: {who} takes {_kind(mine)}, not {_kind(theirs)}."
    if isinstance(mine, (ContinuousDomain, IntegerDomain)):
        if theirs.lower < mine.lower or theirs.upper > mine.upper:
            return (f"{name}: {who} goes from {mine.lower:g} to {mine.upper:g}; keep the "
                    "range inside that.")
    elif isinstance(mine, EnumeratedDomain):
        extra = [v for v in theirs.values if v not in mine.values]
        if extra:
            return (f"{name}: {who} takes {listed(mine.values)}, not "
                    f"{listed(extra)}.")
    elif theirs != mine:  # a simplex: as declared
        return f"{name}: {who} takes it as declared; it can't be changed here."
    limit = getattr(mine, "max_step", None)
    if limit is not None and (theirs.max_step is None or theirs.max_step > limit):
        return (f"{name}: {who} moves it at most {limit:g} per run; keep that travel "
                "limit (or a smaller one).")
    return None


def _tighter(value, bound, pick) -> bool:
    """Whether ``value`` is at least as tight as ``bound`` (``pick`` = max for a
    lower bound, min for an upper one)."""
    if bound is None:
        return True
    return value is not None and pick(value, bound) == value


def _kind(d) -> str:
    return {"continuous": "a range", "integer": "whole numbers", "enumerated": "choices",
            "structured": "fractions summing to 1"}.get(d.kind, d.kind)


def _constraint_words(c) -> str:
    terms = " + ".join(f"{v:g}·{k}" if v != 1 else k for k, v in c.coeffs.items())
    if c.min_val is not None and c.max_val is not None:
        return f"{c.min_val:g} ≤ {terms} ≤ {c.max_val:g}"
    return f"{terms} ≥ {c.min_val:g}" if c.max_val is None else f"{terms} ≤ {c.max_val:g}"


def listed(items) -> str:
    items = [str(i) for i in items]
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _name(target) -> str:
    if isinstance(target, type) or inspect.isfunction(target):
        return target.__name__
    return getattr(target, "name", None) or type(target).__name__


def load(ref: str, spec: str | None = None) -> YourExperiment:
    """``module:Name`` or ``path/to/file.py:Name`` → a :class:`YourExperiment`.

    A module is imported as Python would from the current directory; a file path
    is loaded directly. ``spec`` (``--spec``) is the spec it works with: a JSON file
    (a spec or campaign.json), or ``module:NAME`` / ``file.py:NAME`` naming one;
    without it, the experiment's own ``spec`` attribute, if any. Raises with a
    message naming what's missing."""
    target, shown = _resolve(ref, "--experiment")
    declared = None if spec is None else load_spec(spec)
    return YourExperiment(target, spec=declared, source=shown)


def command(cmd: str, spec: str | None = None, timeout: float | None = None) -> YourExperiment:
    """``--command "./measure.sh {temp} {conc}"`` → a :class:`YourExperiment` that
    runs it for each design (:class:`abbot.experiment.CommandExperiment`), made
    for each campaign's spec. ``spec`` as for :func:`load`."""
    from abbot.experiment import CommandExperiment

    probe = CommandExperiment(cmd, timeout=timeout)  # a bad command fails here
    declared = None
    if spec is not None:
        declared = load_spec(spec)
        probe = CommandExperiment(cmd, spec=declared, timeout=timeout)

    def make(campaign_spec: ProblemSpec) -> CommandExperiment:
        return CommandExperiment(cmd, spec=campaign_spec, timeout=timeout)

    return YourExperiment(make, spec=declared, name=probe.name, source=cmd,
                          reads=probe.fields)


def load_spec(ref: str) -> ProblemSpec:
    """``--spec``: a JSON file (a spec or campaign.json), or ``module:NAME`` /
    ``file.py:NAME`` naming one."""
    if ref.endswith(".json"):
        path = Path(ref).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--spec: no file {path.resolve()}")
        return as_spec(path)
    return as_spec(_resolve(ref, "--spec")[0])


def _resolve(ref: str, flag: str):
    """``module:Name`` or ``file.py:Name`` → (the object, how to show the reference)."""
    module_ref, sep, attr = ref.rpartition(":")
    if not sep or not module_ref or not attr:
        raise ValueError(f"{flag} takes module:Name or file.py:Name, got {ref!r}")
    if module_ref.endswith(".py") or "/" in module_ref:
        path = Path(module_ref).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{flag}: no file {path}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(path.stem, module)
        spec.loader.exec_module(module)
    else:
        if "" not in sys.path and str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))
        module = importlib.import_module(module_ref)
    try:
        target = getattr(module, attr)
    except AttributeError:
        raise AttributeError(f"{flag}: {module_ref} has no {attr!r}") from None
    # A file is named by its name alone on the page: the path is the person's own.
    shown = f"{Path(module_ref).name}:{attr}" if module.__name__ != module_ref else ref
    return target, shown


__all__ = ["YourExperiment", "as_spec", "command", "listed", "load", "load_spec", "within"]
