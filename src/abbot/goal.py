"""Goal ABC — goal-as-algorithm for BAX (DEFERRED, F1).

Unifies multibax ``SubsetAlgorithm`` + nn-bax ``FunctionAlgorithm`` + sbido target
boxes behind one interface. A Goal identifies the subset of the design space that
achieves the user's goal, given either evaluated values (enumerated spaces) or a
callable function sample (continuous spaces). The BAX policy then acquires to
resolve that subset.

Not an M1 deliverable: BAX / goal-as-algorithm is an upstream candidate (contribute
to Xopt) and lands with the BAXstics port (plan F1). The protocol is defined now
so the policy/model seams stay goal-ready; implementations come later.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Goal(Protocol):
    def identify_subspace(
        self, f: torch.Tensor, X: torch.Tensor | None = None
    ) -> list[int] | torch.Tensor:
        """Return the indices / points that satisfy the goal under function sample f."""
        ...
