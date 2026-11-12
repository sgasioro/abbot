"""Your own experiment in the UI: ``--experiment module:Name`` or :func:`abbot.ui.serve`.

The UI runs each design through it, as it runs the built-in simulation. ``Name``
is an experiment (an object with ``run(design)`` or ``run_batch(designs)``,
:class:`abbot.experiment.Experiment`), or a class or function that makes one. A
maker is called when a campaign starts, with the campaign's spec if it takes an
argument, so one experiment class can serve any setup.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from collections.abc import Callable
from pathlib import Path

from abbot.problem import ProblemSpec


def _is_experiment(obj) -> bool:
    return not isinstance(obj, type) and (hasattr(obj, "run") or hasattr(obj, "run_batch"))


class YourExperiment:
    """An experiment for the UI to run: ``make(spec)`` gives the object whose
    ``run`` / ``run_batch`` measures a design; ``evaluate`` (optional) reduces what
    it returns to the outputs. ``name`` labels it on the page, ``source`` says where
    it came from (``rig:Rig``)."""

    def __init__(self, target, *, evaluate: Callable | None = None, name: str | None = None,
                 source: str | None = None) -> None:
        if not _is_experiment(target) and not callable(target):
            raise TypeError(
                f"{source or target!r} is neither an experiment (an object with run(design) "
                "or run_batch(designs)) nor a class or function that makes one."
            )
        self.target = target
        self.evaluate = evaluate
        self.name = name or _name(target)
        self.source = source

    def make(self, spec: ProblemSpec):
        """The experiment for a campaign on ``spec``."""
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


def _name(target) -> str:
    if isinstance(target, type) or inspect.isfunction(target):
        return target.__name__
    return getattr(target, "name", None) or type(target).__name__


def load(ref: str) -> YourExperiment:
    """``module:Name`` or ``path/to/file.py:Name`` → a :class:`YourExperiment`.

    A module is imported as Python would from the current directory; a file path
    is loaded directly. Raises with a message naming what's missing."""
    module_ref, sep, attr = ref.rpartition(":")
    if not sep or not module_ref or not attr:
        raise ValueError(f"--experiment takes module:Name or file.py:Name, got {ref!r}")
    if module_ref.endswith(".py") or "/" in module_ref:
        path = Path(module_ref).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--experiment: no file {path}")
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
        raise AttributeError(f"--experiment: {module_ref} has no {attr!r}") from None
    # A file is named by its name alone on the page: the path is the person's own.
    shown = f"{Path(module_ref).name}:{attr}" if module.__name__ != module_ref else ref
    return YourExperiment(target, source=shown)


__all__ = ["YourExperiment", "load"]
