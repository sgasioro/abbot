"""Runtime registry: name -> class.

Heavy / live objects (a trained CGCNN, an HTTP reactor ``Experiment``) are
referenced from a serializable spec by registry key + pydantic config, not
embedded. This keeps the problem spec and run state round-trippable (frontend +
resume + agents) while still letting a config name a concrete implementation.

Minimal M1 version: a dict-backed registry with a decorator. The pydantic
config schemas that pair a key with its kwargs are added as the frontend lands.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str):
        def deco(cls: type[T]) -> type[T]:
            if name in self._items:
                raise KeyError(f"{self.kind} {name!r} already registered")
            self._items[name] = cls
            return cls

        return deco

    def get(self, name: str) -> type:
        if name not in self._items:
            raise KeyError(
                f"unknown {self.kind} {name!r}; known: {sorted(self._items)}"
            )
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)


POLICIES = Registry("policy")
MODEL_CONSTRUCTORS = Registry("model_constructor")
CANDIDATE_OPTIMIZERS = Registry("candidate_optimizer")
EXPERIMENTS = Registry("experiment")
