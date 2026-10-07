"""ObjectiveReducer — the composite-SBI reduction (M2, stub).

Reduce posterior samples over theta to a scalar/vector objective F. This is the
genuinely-ours composite-SBI layer. The same reducer logic mounts at EITHER of
two points:
  - pre-GP:  reduce to a scalar F(xi) before the surrogate (status quo, simple
             cases) — supply it as the loop's ``evaluate`` callback.
  - in-acqf: keep a multi-output theta surrogate and reduce inside a botorch
             ``MCAcquisitionObjective`` / posterior transform (information-rich
             SBI; the Astudillo-Frazier composite-BO pattern).

Basis: ``~/sbido/src/sbido/objectives.py::selection_score``. Implementation lands
in M2 (the lead application); the protocol is defined now so the loop's
``evaluate`` seam and the model seam are reduction-ready.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from abbot.problem import ProblemSpec


@runtime_checkable
class ObjectiveReducer(Protocol):
    def __call__(
        self, posterior_samples: torch.Tensor, spec: ProblemSpec
    ) -> torch.Tensor:
        """Map (n_samples, d_theta) posterior samples to scalar/vector F."""
        ...
