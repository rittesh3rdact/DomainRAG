"""Falls back through an ordered list of models when one fails -- daily
quota exhausted, a model retired/renamed, or persistent overload -- instead
of hard-failing the whole request. Remembers whichever model last worked so
later calls try it first, rather than re-probing an already-exhausted
primary model on every single call.
"""

import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModelFallback:
    def __init__(self, models: list[str]):
        if not models:
            raise ValueError("models must be a non-empty list")
        self._models = models
        self._preferred = 0

    def call(self, fn: Callable[[str], T]) -> T:
        """fn takes a model name and returns a result, or raises."""
        order = [self._preferred] + [
            i for i in range(len(self._models)) if i != self._preferred
        ]
        last_exc: Exception | None = None
        for idx in order:
            model = self._models[idx]
            try:
                result = fn(model)
                self._preferred = idx
                return result
            except Exception as exc:  # noqa: BLE001 - re-raised once all models are exhausted
                logger.warning(
                    "Model %s failed (%s); falling back to the next model", model, exc
                )
                last_exc = exc
        raise last_exc
