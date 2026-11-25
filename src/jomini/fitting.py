"""Estimating model coefficients from observed battle data."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from jomini.battle import Battle

_MAX_EXPONENT = 2.0


def _as_series(name: str, value: float | ArrayLike) -> NDArray[np.float64]:
    try:
        arr = np.atleast_1d(np.asarray(value, dtype=float))
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number or a sequence of numbers") from None
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a number or a 1-D sequence of numbers")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def trajectory(
    battle: Battle,
    rho: float,
    beta: float,
    p: float,
    q: float,
    t: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Strengths at times ``t`` (shape ``(2, len(t))``), or ``None`` if integration fails."""

    def rhs(_t: float, y: NDArray[np.float64]) -> list[float]:
        red, blue = max(y[0], 0.0), max(y[1], 0.0)
        return [-beta * blue**p * red**q, -rho * red**p * blue**q]

    sol = solve_ivp(
        rhs,
        (0.0, float(t[-1])),
        [battle.red, battle.blue],
        method="LSODA",
        t_eval=t,
        rtol=1e-9,
        atol=1e-9 * max(battle.red, battle.blue),
    )
    if not sol.success or sol.y.shape[1] != t.size:
        return None
    clipped: NDArray[np.float64] = np.clip(sol.y, 0.0, None)
    return clipped


def estimate(
    battle: Battle,
    t: float | Sequence[float] | ArrayLike,
    red: float | Sequence[float] | ArrayLike,
    blue: float | Sequence[float] | ArrayLike,
    *,
    p: float | None,
    q: float | None,
) -> tuple[float, float, float, float]:
    """Least-squares estimate of ``(rho, beta, p, q)``; ``None`` exponents are fitted too."""
    if not isinstance(battle, Battle):
        raise TypeError("battle must be a jomini.Battle")
    ts = _as_series("t", t)
    reds = _as_series("red", red)
    blues = _as_series("blue", blue)
    if not ts.size == reds.size == blues.size:
        raise ValueError("t, red and blue must have the same length")
    if ts[0] <= 0 or np.any(np.diff(ts) <= 0):
        raise ValueError("t must be positive and strictly increasing")
    if np.any(reds < 0) or np.any(reds > battle.red):
        raise ValueError("red observations must lie between 0 and battle.red")
    if np.any(blues < 0) or np.any(blues > battle.blue):
        raise ValueError("blue observations must lie between 0 and battle.blue")

    free_p, free_q = p is None, q is None
    n_free = 2 + free_p + free_q
    if 2 * ts.size < n_free:
        raise ValueError(
            f"fitting {n_free} parameters needs at least {math.ceil(n_free / 2)} observation "
            "times; pass more data or fix p and q"
        )

    # Starting point: exponents at the middle of their usual range, coefficients
    # from the average loss rate over the first observed interval.
    p0 = 0.5 if p is None else p
    q0 = 0.5 if q is None else q
    tiny = 1e-6
    red_rate = max((battle.red - reds[0]) / ts[0], tiny * battle.red / ts[0])
    blue_rate = max((battle.blue - blues[0]) / ts[0], tiny * battle.blue / ts[0])
    beta0 = red_rate / (battle.blue**p0 * battle.red**q0)
    rho0 = blue_rate / (battle.red**p0 * battle.blue**q0)

    x0 = [math.log(rho0), math.log(beta0)]
    lower, upper = [-np.inf, -np.inf], [np.inf, np.inf]
    for free, start in ((free_p, p0), (free_q, q0)):
        if free:
            x0.append(start)
            lower.append(0.0)
            upper.append(_MAX_EXPONENT)

    observed = np.concatenate([reds / battle.red, blues / battle.blue])

    def unpack(x: NDArray[np.float64]) -> tuple[float, float, float, float]:
        rest = iter(x[2:])
        pp = float(next(rest)) if free_p else float(p)  # type: ignore[arg-type]
        qq = float(next(rest)) if free_q else float(q)  # type: ignore[arg-type]
        return math.exp(x[0]), math.exp(x[1]), pp, qq

    def residuals(x: NDArray[np.float64]) -> NDArray[np.float64]:
        y = trajectory(battle, *unpack(x), ts)
        if y is None:
            return np.full(observed.size, 1e3)
        predicted = np.concatenate([y[0] / battle.red, y[1] / battle.blue])
        diff: NDArray[np.float64] = predicted - observed
        return diff

    sol = least_squares(residuals, x0, bounds=(lower, upper), x_scale="jac")
    if not sol.success:
        raise RuntimeError(f"fit did not converge: {sol.message}")
    return unpack(sol.x)
