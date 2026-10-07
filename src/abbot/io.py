"""Import/export helpers for problem specs (JSON) and run history (CSV).

Pure functions over core objects, kept out of the UI so they are testable and
reusable by a notebook or an agent. Spec JSON reuses pydantic's
``model_dump_json`` / ``model_validate_json`` (the serializable contract we already
rely on). History CSV is a flat ``step, <vars...>, <objectives...>`` table — the
BAXstics-style artifact, but spec-driven (columns come from the spec, not
hard-coded).
"""

from __future__ import annotations

import csv
import io

import torch

from abbot.history import History, Record
from abbot.problem import ProblemSpec

# --- ProblemSpec <-> JSON -----------------------------------------------------


def spec_to_json(spec: ProblemSpec) -> str:
    return spec.model_dump_json(indent=2)


def spec_from_json(text: str) -> ProblemSpec:
    return ProblemSpec.model_validate_json(text)


# --- History <-> CSV ----------------------------------------------------------


def history_to_csv(history: History, spec: ProblemSpec) -> str:
    """Flat CSV: ``step, <variable names...>, <objective names...>``.

    Only records with outputs are written. Raw (physical) output units.
    """
    var_names = spec.variable_names
    obj_names = [o.name for o in spec.objectives]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["step", *var_names, *obj_names])
    for i, r in enumerate(history.records):
        if r.outputs is None:
            continue
        xs = r.candidate.reshape(-1).tolist()
        ys = r.outputs.reshape(-1).tolist()
        writer.writerow([i, *xs, *ys])
    return buf.getvalue()


def history_from_csv(text: str, spec: ProblemSpec) -> History:
    """Parse a CSV with columns for each variable + objective into a History.

    Column matching is by name against the spec (a ``step`` column, if present, is
    ignored). Raises a clear error if a required variable/objective column is
    missing. Imported records carry no observation/aux (they are prior data).
    """
    var_names = spec.variable_names
    obj_names = [o.name for o in spec.objectives]
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("empty CSV: no header row")
    missing = [c for c in (*var_names, *obj_names) if c not in reader.fieldnames]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s) {missing}; expected one per variable "
            f"{var_names} and objective {obj_names}."
        )
    history = History()
    for row_num, row in enumerate(reader, start=2):  # row 1 is the header
        try:
            x = torch.tensor([float(row[n]) for n in var_names], dtype=torch.double)
            y = torch.tensor([float(row[n]) for n in obj_names], dtype=torch.double)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CSV row {row_num}: could not parse numbers ({exc})") from exc
        history.append(Record(candidate=x, observation=None, outputs=y))
    return history
