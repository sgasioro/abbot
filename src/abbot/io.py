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
    """Flat CSV: ``step, <variable names...>, <output names...>``.

    One column per modeled output (the full set, incl. OBSERVE), in spec.outputs
    order. Only records with outputs are written. Raw (physical) output units.
    """
    var_names = spec.variable_names
    out_names = spec.output_names
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["step", *var_names, *out_names])
    for i, r in enumerate(history.records):
        if r.outputs is None:
            continue
        xs = r.candidate.reshape(-1).tolist()
        ys = r.outputs.reshape(-1).tolist()
        writer.writerow([i, *xs, *ys])
    return buf.getvalue()


def history_from_csv(text: str, spec: ProblemSpec) -> History:
    """Parse a CSV with columns for each variable + output into a History.

    Column matching is by name against the spec (a ``step`` column, if present, is
    ignored). Raises a clear error if a required variable/output column is missing.
    Imported records carry no observation/aux (they are prior data).
    """
    var_names = spec.variable_names
    out_names = spec.output_names
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("empty CSV: no header row")
    missing = [c for c in (*var_names, *out_names) if c not in reader.fieldnames]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s) {missing}; expected one per variable "
            f"{var_names} and output {out_names}."
        )
    history = History()
    for row_num, row in enumerate(reader, start=2):  # row 1 is the header
        try:
            x = torch.tensor([float(row[n]) for n in var_names], dtype=torch.double)
            y = torch.tensor([float(row[n]) for n in out_names], dtype=torch.double)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CSV row {row_num}: could not parse numbers ({exc})") from exc
        history.append(Record(candidate=x, observation=None, outputs=y))
    return history


# --- Self-describing CSV: spec + history in one file --------------------------
# The ``#``-prefixed header rows encode the full ProblemSpec; the measurement
# table follows. Config lives in comment rows, so pandas (`read_csv(comment='#')`)
# and Excel read the data table directly — one portable file carries setup + data.
# Adopted from the bax-platform student tool; abbot adds a version tag + domain
# types (continuous/enumerated) + boxes. Row grammar (each ``# key,value...``):
#   # version,1
#   # continuous,<name>,<lower>,<upper>
#   # integer,<name>,<lower>,<upper>
#   # enumerated,<name>,<v1>,<v2>,...
#   # output,<name>,MINIMIZE|MAXIMIZE|OBSERVE[,<min_val>,<max_val>]
#                                     (OBSERVE = modeled, not optimized; a bound
#                                     left empty is absent)
#   # box,<lo1>,<lo2>,...,|,<hi1>,<hi2>,...     (lower vector | upper vector)
# Legacy ``# objective,<name>,<DIR>`` rows are still READ (mapped to an output).
# NOT representable here (raises on write): StructuredDomain, domain constraints,
# output constraints over several outputs, source variables, cached aux. Use the JSON spec / DataStore for those.

CSV_FORMAT_VERSION = 1


def spec_history_to_csv(spec: ProblemSpec, history: History | None = None) -> str:
    """Write a self-describing CSV: ``#`` config rows + the measurement table.

    Raises ``ValueError`` if the spec has parts CSV can't hold (structured domains,
    domain constraints, output constraints over several outputs) — be honest rather than silently dropping them; use the
    JSON spec for those campaigns.
    """
    from abbot.problem import ContinuousDomain, EnumeratedDomain, IntegerDomain

    if spec.sources:
        raise ValueError(
            "self-describing CSV cannot represent source variables "
            f"{spec.source_names}; use the JSON spec (spec_to_json) for this campaign."
        )
    if spec.domain_constraints:
        raise ValueError(
            "self-describing CSV cannot represent inequality constraints; use the "
            "JSON spec (spec_to_json) for this campaign."
        )
    if spec.output_constraints:
        raise ValueError(
            "self-describing CSV cannot represent output constraints over several "
            "outputs; use the JSON spec (spec_to_json) for this campaign."
        )
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["# version", CSV_FORMAT_VERSION])
    for d in spec.domain:
        if isinstance(d, ContinuousDomain):
            w.writerow(["# continuous", d.name, d.lower, d.upper])
        elif isinstance(d, IntegerDomain):
            w.writerow(["# integer", d.name, d.lower, d.upper])
        elif isinstance(d, EnumeratedDomain):
            w.writerow(["# enumerated", d.name, *d.values])
        else:
            raise ValueError(  # noqa: TRY004 - a format-capability limit, not a type error
                f"self-describing CSV cannot represent {type(d).__name__} "
                f"({d.name!r}); use the JSON spec for this campaign."
            )
    for o in spec.outputs:
        bounds = [] if not o.is_constrained else [
            "" if v is None else v for v in (o.min_val, o.max_val)
        ]
        w.writerow(["# output", o.name, o.direction, *bounds])
    for box in spec.boxes or []:
        w.writerow(["# box", *box.lower, "|", *box.upper])
    # Measurement table (reuse the flat writer's columns = all outputs).
    if history is not None and len(history):
        buf.write(history_to_csv(history, spec))
    else:
        w.writerow(["step", *spec.variable_names, *spec.output_names])
    return buf.getvalue()


def spec_history_from_csv(text: str) -> tuple[ProblemSpec, History]:
    """Parse a self-describing CSV back into ``(ProblemSpec, History)``.

    Consumes leading ``#`` config rows to rebuild the spec, then parses the
    measurement table. Unknown ``#`` keys are ignored (treated as comments).
    """
    from abbot.problem import (
        Box,
        ContinuousDomain,
        EnumeratedDomain,
        IntegerDomain,
        OutputSpec,
    )

    config_lines: list[str] = []
    data_lines: list[str] = []
    for line in text.splitlines():
        (config_lines if line.lstrip().startswith("#") else data_lines).append(line)

    domain: list = []
    outputs: list = []
    boxes: list = []
    for raw in config_lines:
        cells = [c.strip() for c in raw.lstrip().lstrip("#").strip().split(",")]
        key = cells[0].lower()
        if key == "continuous":
            if len(cells) < 4:
                raise ValueError(
                    f"self-describing CSV: malformed `# continuous` row {raw!r}; "
                    "expected `# continuous,<name>,<lower>,<upper>`."
                )
            domain.append(ContinuousDomain(name=cells[1], lower=float(cells[2]), upper=float(cells[3])))
        elif key == "integer":
            if len(cells) < 4:
                raise ValueError(
                    f"self-describing CSV: malformed `# integer` row {raw!r}; "
                    "expected `# integer,<name>,<lower>,<upper>`."
                )
            domain.append(IntegerDomain(name=cells[1], lower=int(cells[2]), upper=int(cells[3])))
        elif key == "enumerated":
            if len(cells) < 3:
                raise ValueError(
                    f"self-describing CSV: malformed `# enumerated` row {raw!r}; "
                    "expected `# enumerated,<name>,<v1>,<v2>,...`."
                )
            domain.append(EnumeratedDomain(name=cells[1], values=[float(v) for v in cells[2:] if v]))
        elif key in ("output", "objective"):  # "objective" = legacy read-compat
            direction = cells[2].upper() if len(cells) > 2 else "MAXIMIZE"
            bounds = [float(v) if v else None for v in (cells[3:5] + ["", ""])[:2]]
            outputs.append(OutputSpec(
                name=cells[1], direction=direction, min_val=bounds[0], max_val=bounds[1]
            ))
        elif key == "box":
            if "|" not in cells:
                raise ValueError(
                    f"self-describing CSV: malformed `# box` row {raw!r}; expected a "
                    "`|` separating the lower and upper vectors "
                    "(`# box,<lo1>,...,|,<hi1>,...`)."
                )
            sep = cells.index("|")
            lower = [float(v) for v in cells[1:sep]]
            upper = [float(v) for v in cells[sep + 1:] if v]
            boxes.append(Box(lower=lower, upper=upper))
        # version / unknown keys: ignored (comments)
    if not domain or not outputs:
        raise ValueError(
            "self-describing CSV missing `# continuous`/`# integer`/`# enumerated` and/or "
            "`# output` config rows; cannot rebuild the ProblemSpec."
        )
    spec = ProblemSpec(domain=domain, outputs=outputs, boxes=boxes or None)
    history = history_from_csv("\n".join(data_lines), spec)
    return spec, history
