"""Models for modern combat: salvo exchanges and mixed forces.

* :class:`SalvoModel`: Hughes' (1995) salvo equations, for fights decided by
  discrete volleys of missiles or drones against interceptors (naval combat, drone
  swarms against air defence).
* :class:`MixedForceLaw`: heterogeneous Lanchester combat between sides made of
  several unit types (tanks, infantry, drones, ...) with a kill-rate matrix between
  every pair of types.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

# --------------------------------------------------------------------------- salvo model


@dataclass(frozen=True, slots=True)
class SalvoSide:
    """One side's per-unit parameters in Hughes' salvo model."""

    offense: float
    """Well-aimed shots (missiles, drones) each unit fires per salvo."""
    defense: float
    """Incoming shots each unit can intercept per salvo."""
    staying_power: float
    """Hits needed to put one unit out of action."""

    def __post_init__(self) -> None:
        for name in ("offense", "defense"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be a non-negative number")
        if not (math.isfinite(self.staying_power) and self.staying_power > 0):
            raise ValueError("staying_power must be positive")


@dataclass(frozen=True)
class SalvoResult:
    red: NDArray[np.float64]
    """Red's surviving units before salvo 0, 1, 2, ... and after the last."""
    blue: NDArray[np.float64]

    @property
    def salvos(self) -> int:
        return int(self.red.size - 1)

    @property
    def remaining(self) -> tuple[float, float]:
        return float(self.red[-1]), float(self.blue[-1])

    @property
    def winner(self) -> Literal["red", "blue"] | None:
        r, b = self.remaining
        if r <= 0 < b:
            return "blue"
        if b <= 0 < r:
            return "red"
        return None


@dataclass(frozen=True)
class SalvoModel:
    """Hughes' salvo equations (simultaneous fire)::

        blue_lost = max(0, red_offense * R - blue_defense * B) / blue_staying_power
        red_lost  = max(0, blue_offense * B - red_defense * R) / red_staying_power

    Unlike Lanchester's continuous fire, a salvo can be absorbed entirely by
    defences. Once offence outruns defence, losses jump in one volley, which is
    why the model favours striking first and in mass.
    """

    red: SalvoSide
    blue: SalvoSide

    def exchange(self, red: float, blue: float) -> tuple[float, float]:
        """Units ``(red, blue)`` lost in one simultaneous salvo exchange."""
        if red < 0 or blue < 0:
            raise ValueError("unit counts must be non-negative")
        blue_lost = (
            max(0.0, self.red.offense * red - self.blue.defense * blue) / self.blue.staying_power
        )
        red_lost = (
            max(0.0, self.blue.offense * blue - self.red.defense * red) / self.red.staying_power
        )
        return min(red_lost, red), min(blue_lost, blue)

    def fractional_exchange_ratio(self, red: float, blue: float) -> float:
        """Hughes' FER: blue's fractional loss divided by red's (>1 favours red)."""
        red_lost, blue_lost = self.exchange(red, blue)
        if red_lost == 0:
            return math.inf if blue_lost > 0 else math.nan
        return (blue_lost / blue) / (red_lost / red)

    def simulate(
        self,
        red: float,
        blue: float,
        *,
        salvos: int = 10,
        first: Literal["both", "red", "blue"] = "both",
    ) -> SalvoResult:
        """Repeat exchanges until one side is gone or ``salvos`` have been fired.

        ``first`` lets one side fire an opening salvo unanswered (a surprise strike).
        """
        if salvos < 1:
            raise ValueError("salvos must be at least 1")
        r, b = float(red), float(blue)
        if r < 0 or b < 0:
            raise ValueError("unit counts must be non-negative")
        reds, blues = [r], [b]
        if first in ("red", "blue"):
            red_lost, blue_lost = self.exchange(r, b)
            if first == "red":
                b -= blue_lost
            else:
                r -= red_lost
            reds.append(r)
            blues.append(b)
            salvos -= 1
        elif first != "both":
            raise ValueError("first must be 'both', 'red' or 'blue'")
        for _ in range(salvos):
            if r <= 0 or b <= 0:
                break
            red_lost, blue_lost = self.exchange(r, b)
            r, b = max(r - red_lost, 0.0), max(b - blue_lost, 0.0)
            reds.append(r)
            blues.append(b)
        return SalvoResult(np.array(reds), np.array(blues))


# --------------------------------------------------------------------------- mixed forces


Allocation = Literal["proportional", "uniform"]


@dataclass(frozen=True)
class MixedForceResult:
    t: NDArray[np.float64]
    red: dict[str, NDArray[np.float64]]
    blue: dict[str, NDArray[np.float64]]
    loser: Literal["red", "blue"] | None

    @property
    def winner(self) -> Literal["red", "blue"] | None:
        if self.loser is None:
            return None
        return "blue" if self.loser == "red" else "red"

    @property
    def remaining(self) -> tuple[dict[str, float], dict[str, float]]:
        return (
            {k: float(v[-1]) for k, v in self.red.items()},
            {k: float(v[-1]) for k, v in self.blue.items()},
        )


@dataclass(frozen=True)
class MixedForceLaw:
    """Heterogeneous aimed-fire (square-law) combat between multi-type forces.

    ``red_kills[i][j]`` is how many blue units of type ``j`` one red unit of type ``i``
    kills per unit time when firing only at type ``j``; ``blue_kills`` likewise.
    Each shooter splits its fire across enemy types by ``allocation``:
    ``"proportional"`` (in proportion to the enemy's surviving numbers, i.e. it
    shoots at what is there) or ``"uniform"`` (equally among surviving types).
    """

    red_types: tuple[str, ...]
    blue_types: tuple[str, ...]
    red_kills: NDArray[np.float64] = field(repr=False)
    blue_kills: NDArray[np.float64] = field(repr=False)
    allocation: Allocation = "proportional"

    @classmethod
    def from_rates(
        cls,
        red_kills: Mapping[str, Mapping[str, float]],
        blue_kills: Mapping[str, Mapping[str, float]],
        allocation: Allocation = "proportional",
    ) -> MixedForceLaw:
        """Build from nested dicts, e.g. ``{"drone": {"tank": 0.05, "infantry": 0.2}}``."""
        red_types = tuple(red_kills)
        blue_types = tuple(blue_kills)
        rk = np.array([[red_kills[i].get(j, 0.0) for j in blue_types] for i in red_types], float)
        bk = np.array([[blue_kills[i].get(j, 0.0) for j in red_types] for i in blue_types], float)
        for side, kills in (("red", red_kills), ("blue", blue_kills)):
            other = blue_types if side == "red" else red_types
            for shooter, targets in kills.items():
                unknown = set(targets) - set(other)
                if unknown:
                    raise ValueError(f"{side} {shooter!r} targets unknown types {sorted(unknown)}")
        return cls(red_types, blue_types, rk, bk, allocation)

    def __post_init__(self) -> None:
        if self.allocation not in ("proportional", "uniform"):
            raise ValueError("allocation must be 'proportional' or 'uniform'")
        if self.red_kills.shape != (len(self.red_types), len(self.blue_types)):
            raise ValueError("red_kills must be shaped (red types, blue types)")
        if self.blue_kills.shape != (len(self.blue_types), len(self.red_types)):
            raise ValueError("blue_kills must be shaped (blue types, red types)")
        if np.any(~np.isfinite(self.red_kills)) or np.any(self.red_kills < 0):
            raise ValueError("kill rates must be non-negative numbers")
        if np.any(~np.isfinite(self.blue_kills)) or np.any(self.blue_kills < 0):
            raise ValueError("kill rates must be non-negative numbers")

    def _weights(
        self, targets: NDArray[np.float64], kills: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Fire allocation matrix (shooter types x target types), rows summing to 1."""
        alive = (targets > 1e-9) & (kills > 0)  # only shoot at reachable, surviving types
        if self.allocation == "proportional":
            w = np.where(alive, targets[None, :], 0.0)
        else:
            w = alive.astype(float)
        total = w.sum(axis=1, keepdims=True)
        out: NDArray[np.float64] = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
        return out

    def rates(
        self, red: ArrayLike, blue: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Loss rates per type ``(red_losses, blue_losses)`` at the given strengths."""
        r = np.clip(np.asarray(red, float), 0, None)
        b = np.clip(np.asarray(blue, float), 0, None)
        blue_losses = (r[:, None] * self.red_kills * self._weights(b, self.red_kills)).sum(axis=0)
        red_losses = (b[:, None] * self.blue_kills * self._weights(r, self.blue_kills)).sum(axis=0)
        return red_losses, blue_losses

    def simulate(
        self,
        red: Mapping[str, float] | Sequence[float],
        blue: Mapping[str, float] | Sequence[float],
        *,
        t_max: float,
        breakpoint: float | None = None,
        n_points: int = 201,
    ) -> MixedForceResult:
        """Integrate until ``t_max`` or until a side's *total* losses reach ``breakpoint``."""
        r0 = self._vector(red, self.red_types, "red")
        b0 = self._vector(blue, self.blue_types, "blue")
        if not (math.isfinite(t_max) and t_max > 0):
            raise ValueError("t_max must be positive")
        if breakpoint is not None and not 0 < breakpoint <= 1:
            raise ValueError("breakpoint must be in (0, 1]")
        nr = len(self.red_types)
        bp = 1.0 if breakpoint is None else breakpoint
        floor_r, floor_b = (1 - bp) * r0.sum() + 1e-9, (1 - bp) * b0.sum() + 1e-9

        def rhs(_t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            red_l, blue_l = self.rates(y[:nr], y[nr:])
            return np.concatenate([-red_l, -blue_l])

        def red_down(_t: float, y: NDArray[np.float64]) -> float:
            return float(np.clip(y[:nr], 0, None).sum() - floor_r)

        def blue_down(_t: float, y: NDArray[np.float64]) -> float:
            return float(np.clip(y[nr:], 0, None).sum() - floor_b)

        for ev in (red_down, blue_down):
            ev.terminal = True  # type: ignore[attr-defined]
            ev.direction = -1  # type: ignore[attr-defined]
        sol = solve_ivp(
            rhs,
            (0, t_max),
            np.concatenate([r0, b0]),
            method="LSODA",
            events=(red_down, blue_down),
            dense_output=True,
            rtol=1e-8,
            atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError(f"integration failed: {sol.message}")
        t_r = sol.t_events[0][0] if sol.t_events[0].size else math.inf
        t_b = sol.t_events[1][0] if sol.t_events[1].size else math.inf
        end = min(t_r, t_b, t_max)
        loser: Literal["red", "blue"] | None = None
        if math.isfinite(min(t_r, t_b)):
            loser = "red" if t_r <= t_b else "blue"
        t = np.linspace(0, end, n_points)
        y = np.clip(sol.sol(t), 0, None)
        return MixedForceResult(
            t=t,
            red={k: y[i] for i, k in enumerate(self.red_types)},
            blue={k: y[nr + j] for j, k in enumerate(self.blue_types)},
            loser=loser,
        )

    @staticmethod
    def _vector(
        values: Mapping[str, float] | Sequence[float], types: tuple[str, ...], side: str
    ) -> NDArray[np.float64]:
        if isinstance(values, Mapping):
            unknown = set(values) - set(types)
            if unknown:
                raise ValueError(f"unknown {side} types {sorted(unknown)}")
            vec = np.array([float(values.get(k, 0.0)) for k in types])
        else:
            vec = np.asarray(values, float)
            if vec.shape != (len(types),):
                raise ValueError(f"{side} needs {len(types)} strengths")
        if np.any(vec < 0) or not np.all(np.isfinite(vec)) or vec.sum() <= 0:
            raise ValueError(f"{side} strengths must be non-negative with a positive total")
        return vec
