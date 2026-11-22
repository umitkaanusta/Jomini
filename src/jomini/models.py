"""Lanchester-type attrition models.

Every model here is a special case of Bracken's generalised Lanchester law::

    dR/dt = -beta * B**p * R**q
    dB/dt = -rho  * R**p * B**q

where ``R`` and ``B`` are the red and blue strengths, ``rho`` is red's
effectiveness (how fast red kills blue) and ``beta`` is blue's effectiveness.

=================  ===  ===
Law                 p    q
=================  ===  ===
Square (aimed)      1    0
Linear (area fire)  1    1
Logarithmic         0    1
=================  ===  ===
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from jomini.battle import Battle
from jomini.result import BattleResult, Outcome, Side

# Integration horizon (in units of the battle's characteristic time) used when
# no t_max is given. Large enough for slow-decaying laws to reach min_strength.
_HORIZON = 1e12


def _check_positive(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")


def _check_exponent(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number, got {value}")


@dataclass(frozen=True)
class LanchesterModel(ABC):
    """Base class: a pair of effectiveness coefficients plus the law's exponents."""

    rho: float
    """Red's effectiveness: blue casualties per unit time per unit of red fire."""
    beta: float
    """Blue's effectiveness: red casualties per unit time per unit of blue fire."""

    def __post_init__(self) -> None:
        _check_positive("rho", self.rho)
        _check_positive("beta", self.beta)

    @property
    @abstractmethod
    def exponents(self) -> tuple[float, float]:
        """The law's ``(p, q)`` exponents."""

    def casualty_rates(self, red: float, blue: float) -> tuple[float, float]:
        """Instantaneous casualty rates ``(red_losses, blue_losses)`` per unit time."""
        p, q = self.exponents
        red, blue = max(red, 0.0), max(blue, 0.0)
        return self.beta * blue**p * red**q, self.rho * red**p * blue**q

    def simulate(
        self,
        battle: Battle,
        *,
        t_max: float | None = None,
        breakpoint: float | tuple[float, float] | None = None,
        min_strength: float = 1.0,
        n_points: int = 201,
    ) -> BattleResult:
        """Integrate the model from ``battle``'s starting strengths.

        :param t_max: Stop after this much time even if no side has lost.
            If ``None``, fight until a side is destroyed or breaks.
        :param breakpoint: Fraction of its initial strength a side can lose
            before it breaks and retreats, e.g. ``0.3``. Pass a ``(red, blue)``
            tuple for different values per side. ``None`` means fight to the end.
        :param min_strength: A side below this strength counts as destroyed.
            Needed because the linear and logarithmic laws only approach zero
            asymptotically.
        :param n_points: Number of samples in the returned trajectory.
        """
        if not isinstance(battle, Battle):
            raise TypeError("battle must be a jomini.Battle")
        if t_max is not None:
            _check_positive("t_max", t_max)
        if isinstance(min_strength, bool) or not isinstance(min_strength, (int, float)):
            raise TypeError("min_strength must be a number")
        if not 0 <= min_strength < min(battle.red, battle.blue):
            raise ValueError("min_strength must be >= 0 and below both sides' initial strength")
        if not isinstance(n_points, int) or n_points < 2:
            raise ValueError("n_points must be an integer >= 2")

        bp_red, bp_blue = _normalise_breakpoint(breakpoint)
        floor_red = max(min_strength, (1 - bp_red) * battle.red)
        floor_blue = max(min_strength, (1 - bp_blue) * battle.blue)
        broke_red = floor_red > min_strength
        broke_blue = floor_blue > min_strength

        def rhs(_t: float, y: NDArray[np.float64]) -> list[float]:
            red_loss, blue_loss = self.casualty_rates(y[0], y[1])
            return [-red_loss, -blue_loss]

        def red_down(_t: float, y: NDArray[np.float64]) -> float:
            return float(y[0] - floor_red)

        def blue_down(_t: float, y: NDArray[np.float64]) -> float:
            return float(y[1] - floor_blue)

        for event in (red_down, blue_down):
            event.terminal = True  # type: ignore[attr-defined]
            event.direction = -1  # type: ignore[attr-defined]

        if t_max is None:
            red_loss, blue_loss = self.casualty_rates(battle.red, battle.blue)
            timescale = min(battle.red / red_loss, battle.blue / blue_loss)
            t_end = _HORIZON * timescale
        else:
            t_end = float(t_max)

        sol = solve_ivp(
            rhs,
            (0.0, t_end),
            [float(battle.red), float(battle.blue)],
            method="LSODA",
            events=(red_down, blue_down),
            dense_output=True,
            rtol=1e-10,
            atol=1e-10 * max(battle.red, battle.blue),
        )
        if not sol.success:
            raise RuntimeError(f"integration failed: {sol.message}")

        t_red = sol.t_events[0][0] if sol.t_events[0].size else math.inf
        t_blue = sol.t_events[1][0] if sol.t_events[1].size else math.inf
        loser: Side | None
        if math.isinf(t_red) and math.isinf(t_blue):
            if t_max is None:
                raise RuntimeError(
                    "battle did not end within the integration horizon; "
                    "raise min_strength, set a breakpoint, or pass t_max"
                )
            loser, outcome, duration = None, Outcome.TIME_LIMIT, t_end
        else:
            loser = "red" if t_red <= t_blue else "blue"
            duration = min(t_red, t_blue)
            broke = broke_red if loser == "red" else broke_blue
            outcome = Outcome.BREAKPOINT if broke else Outcome.ANNIHILATION

        t = np.linspace(0.0, duration, n_points)
        y = np.clip(sol.sol(t), 0.0, None)
        return BattleResult(t=t, red=y[0], blue=y[1], outcome=outcome, loser=loser, _dense=sol.sol)


def _normalise_breakpoint(
    breakpoint: float | tuple[float, float] | None,
) -> tuple[float, float]:
    if breakpoint is None:
        return 1.0, 1.0
    pair = breakpoint if isinstance(breakpoint, tuple) else (breakpoint, breakpoint)
    if len(pair) != 2:
        raise ValueError("breakpoint tuple must be (red, blue)")
    for value in pair:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("breakpoint must be a number or a (red, blue) tuple of numbers")
        if not 0 < value <= 1:
            raise ValueError("breakpoint must be in (0, 1]")
    return float(pair[0]), float(pair[1])


@dataclass(frozen=True)
class SquareLaw(LanchesterModel):
    """Aimed fire: each side's losses scale with the *enemy's* size.

    Fighting strength is proportional to the square of numbers, so
    concentration of force pays off quadratically. Suits musketry, Napoleonic
    line battles and modern direct fire.
    """

    @property
    def exponents(self) -> tuple[float, float]:
        return 1.0, 0.0


@dataclass(frozen=True)
class LinearLaw(LanchesterModel):
    """Unaimed / area fire: losses scale with the product of both sides' sizes.

    Fire is aimed at an area, so more targets in it means more hits. Suits
    artillery and other indirect fire. Also the classic model for melee
    when read as many simultaneous one-on-one duels along a front.
    """

    @property
    def exponents(self) -> tuple[float, float]:
        return 1.0, 1.0


@dataclass(frozen=True)
class LogarithmicLaw(LanchesterModel):
    """Losses scale with a side's *own* size (Peterson; Weiss's Civil War analysis).

    Attrition comes mostly from exposure (disease, fatigue, non-combat losses)
    rather than enemy fire. Has also been used for large-scale tank battles.
    """

    @property
    def exponents(self) -> tuple[float, float]:
        return 0.0, 1.0


@dataclass(frozen=True)
class GeneralLaw(LanchesterModel):
    """Bracken's generalised law with free exponents ``p`` and ``q``.

    ``(p, q) = (1, 0)`` is the square law, ``(1, 1)`` the linear law and
    ``(0, 1)`` the logarithmic law. Fitted values for real battles usually
    fall in between.
    """

    p: float = 1.0
    q: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_exponent("p", self.p)
        _check_exponent("q", self.q)

    @property
    def exponents(self) -> tuple[float, float]:
        return float(self.p), float(self.q)
