"""Goal-as-algorithm layer for BAX (F1).

A ``Goal`` identifies the subset of a candidate grid that achieves the user's
goal, given per-candidate predicted outputs ``f`` of shape ``(N, d_y)`` in RAW
objective units. It returns a list of integer indices into the grid (not a mask,
not points) — the contract the BAX acquisition executes on each sampled function.

Ports multibax-sklearn's subset algorithms (``algorithms.py``) onto abbot: abbot
has no sklearn scalers, so ``f`` arrives already in raw units (the ``BAXPolicy``
converts the canonical-space model output before calling the goal, mirroring
multibax's ``SubsetAlgorithm.unnormalize``). ``PercentileSet`` + ``Wishlist`` cover
both sbido BAX designers; the richer multibax algorithms (Pareto, etc.) port on
demand.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from abbot.problem import ProblemSpec


@runtime_checkable
class Goal(Protocol):
    def identify_subspace(
        self, f: torch.Tensor, X: torch.Tensor | None = None
    ) -> list[int]:
        """Indices of candidates (rows of ``f``) that satisfy the goal.

        ``f``: (N, d_y) predicted outputs in raw units. ``X``: (N, d_x) candidates
        (unused by band/percentile goals; available for x-dependent goals).
        """
        ...


class PercentileSet:
    """Top ``(100 - p)``% of candidates by a single-output score (``PercentileSet1D``).

    Data-relative: the threshold is recomputed from ``f`` on every call, so under
    BAX it differs per posterior sample. Single-output only.
    """

    def __init__(self, percentile: float = 80.0) -> None:
        self.percentile = percentile

    def identify_subspace(
        self, f: torch.Tensor, X: torch.Tensor | None = None
    ) -> list[int]:
        f = f.reshape(f.shape[0], -1)
        if f.shape[1] != 1:
            raise ValueError(
                f"PercentileSet is single-output but f has {f.shape[1]} columns; "
                "use Wishlist/BandIntersection for multi-output targets."
            )
        col = f[:, 0]
        thresh = torch.quantile(col, self.percentile / 100.0)
        return (col >= thresh).nonzero(as_tuple=True)[0].tolist()


class BandIntersection:
    """Candidates inside a per-output ``[lo, hi]`` box on EVERY output.

    ``bounds``: list of ``(lo, hi)`` per output (``multi_level_region_intersection_Nd``).
    """

    def __init__(self, bounds: list[tuple[float, float]]) -> None:
        self.bounds = bounds

    def identify_subspace(
        self, f: torch.Tensor, X: torch.Tensor | None = None
    ) -> list[int]:
        return _band_indices(f, self.bounds)


class Wishlist:
    """Union of ``BandIntersection`` boxes — OR of AND-boxes (``Wishlist``).

    ``bounds_list``: list of per-output ``[(lo, hi), ...]`` specs; a candidate is in
    the subset if it satisfies ANY one box (on all that box's outputs).
    """

    def __init__(self, bounds_list: list[list[tuple[float, float]]]) -> None:
        self.bounds_list = bounds_list

    def identify_subspace(
        self, f: torch.Tensor, X: torch.Tensor | None = None
    ) -> list[int]:
        ids: set[int] = set()
        for bounds in self.bounds_list:
            ids.update(_band_indices(f, bounds))
        return sorted(ids)


def _band_indices(f: torch.Tensor, bounds: list[tuple[float, float]]) -> list[int]:
    """Rows of ``f`` inside ``[lo, hi]`` on every listed output dimension."""
    f = f.reshape(f.shape[0], -1)
    if len(bounds) != f.shape[1]:
        raise ValueError(
            f"bounds has {len(bounds)} entries but f has {f.shape[1]} outputs."
        )
    mask = torch.ones(f.shape[0], dtype=torch.bool)
    for j, (lo, hi) in enumerate(bounds):
        mask &= (f[:, j] >= lo) & (f[:, j] <= hi)
    return mask.nonzero(as_tuple=True)[0].tolist()


def goals_from_spec(spec: ProblemSpec) -> Wishlist:
    """Build a ``Wishlist`` from a ``ProblemSpec``'s target boxes (OR over boxes).

    Each ``Box`` becomes one AND-across-outputs band; the Wishlist unions them —
    the sbido multi-box OR target maps straight in. Requires ``spec.boxes``.
    """
    if not spec.boxes:
        raise ValueError("goals_from_spec requires ProblemSpec.boxes (target bands).")
    bounds_list = [
        [(float(lo), float(hi)) for lo, hi in zip(box.lower, box.upper)]
        for box in spec.boxes
    ]
    return Wishlist(bounds_list)
