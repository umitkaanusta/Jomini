"""Initial conditions of a two-sided engagement."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Battle:
    """Starting strengths of the two sides.

    Strengths are floats so that forces can be expressed in any unit
    (soldiers, companies, tanks, normalised strength, ...).
    """

    red: float
    blue: float

    def __post_init__(self) -> None:
        for side in ("red", "blue"):
            value = getattr(self, side)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{side} must be a number, got {type(value).__name__}")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{side} must be a positive finite number, got {value}")
