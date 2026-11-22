"""Simulation output."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable

Side = Literal["red", "blue"]


class Outcome(str, Enum):
    """Why the simulation stopped."""

    ANNIHILATION = "annihilation"
    """A side fell below ``min_strength``."""
    BREAKPOINT = "breakpoint"
    """A side lost its ``breakpoint`` fraction of men and broke."""
    TIME_LIMIT = "time_limit"
    """``t_max`` was reached with both sides still fighting."""


@dataclass(frozen=True)
class BattleResult:
    """Trajectory and outcome of a simulated battle.

    ``t``, ``red`` and ``blue`` are sampled on an evenly spaced grid from 0 to
    ``duration``; use :meth:`at` for the exact state at any other time.
    """

    t: NDArray[np.float64]
    red: NDArray[np.float64]
    blue: NDArray[np.float64]
    outcome: Outcome
    loser: Side | None
    """The side that was destroyed or broke first, or ``None`` on a time limit."""
    _dense: Callable[[Any], NDArray[np.float64]] = field(repr=False, compare=False)

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    @property
    def winner(self) -> Side | None:
        if self.loser is None:
            return None
        return "blue" if self.loser == "red" else "red"

    @property
    def initial(self) -> tuple[float, float]:
        return float(self.red[0]), float(self.blue[0])

    @property
    def remaining(self) -> tuple[float, float]:
        return float(self.red[-1]), float(self.blue[-1])

    @property
    def casualties(self) -> tuple[float, float]:
        (r0, b0), (r1, b1) = self.initial, self.remaining
        return r0 - r1, b0 - b1

    def at(self, time: float) -> tuple[float, float]:
        """Strengths ``(red, blue)`` at ``time`` (clamped to the battle's duration)."""
        if time < 0:
            raise ValueError("time must be non-negative")
        state = self._dense(min(time, self.duration))
        return max(float(state[0]), 0.0), max(float(state[1]), 0.0)

    def to_pandas(self) -> Any:
        """Trajectory as a ``pandas.DataFrame`` indexed by time (needs ``jomini[pandas]``)."""
        import pandas as pd

        return pd.DataFrame({"red": self.red, "blue": self.blue}, index=pd.Index(self.t, name="t"))

    def plot(self, ax: Any = None) -> Any:
        """Plot both sides' strength over time (needs ``jomini[plot]``). Returns the axes."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.t, self.red, color="tab:red", label="Red")
        ax.plot(self.t, self.blue, color="tab:blue", label="Blue")
        ax.set_xlabel("Time")
        ax.set_ylabel("Strength")
        ax.set_ylim(bottom=0)
        ax.legend()
        return ax

    def __str__(self) -> str:
        (r0, b0), (r1, b1) = self.initial, self.remaining
        verdict = f"{self.winner} wins by {self.outcome.value}" if self.winner else "undecided"
        return (
            f"Battle lasted {self.duration:.4g} time units ({verdict})\n"
            f"Red:  {r0:,.0f} -> {r1:,.0f}  (casualties {r0 - r1:,.0f})\n"
            f"Blue: {b0:,.0f} -> {b1:,.0f}  (casualties {b0 - b1:,.0f})"
        )
