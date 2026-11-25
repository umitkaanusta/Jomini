"""Time-varying attrition: state-space models with Kalman filtering and smoothing.

Classic Lanchester models assume each side's effectiveness is constant. In long
wars it isn't: tactics, weapons (drones), morale and supply change it. This module
treats the quantity of interest as a hidden random walk and recovers its path, with
uncertainty, from noisy loss counts.

* :func:`effectiveness_over_time`: red's per-unit effectiveness relative to blue's,
  ``log(rho/beta)``, period by period. Under Bracken's law the loss ratio satisfies
  ``lossR/lossB = (beta/rho) * (B/R)**(p-q)``, so each period's losses (and, unless
  the law is linear, strengths) give a noisy reading of ``log(rho/beta)``. Under
  the linear law (``exponent=1``, what history follows once quality is accounted
  for) strengths cancel, so loss counts alone suffice.
* :func:`intensity_over_time`: the underlying rate of a single loss series (e.g.
  daily personnel losses), separating trend from day-to-day noise.

Observation noise is Poisson-like (a period with few losses says less) plus an
estimated overdispersion term; the random walk's step size is estimated by
maximum likelihood. Smoothing uses the Rauch-Tung-Striebel recursions.

From a single series the step size and overdispersion are only weakly separable
(fast real change and noisy readings look alike), so treat those two estimates as
rough; the smoothed path and its band are robust. Fix either one if you know it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass(frozen=True)
class StatePath:
    """Smoothed path of a hidden random walk, on the log scale."""

    t: NDArray[np.float64]
    mean: NDArray[np.float64]
    sd: NDArray[np.float64]
    observed: NDArray[np.float64]
    """The noisy per-period readings the path was smoothed from."""
    step_sd: float
    """Estimated SD of the random walk's change per unit time."""
    overdispersion_sd: float
    """Estimated observation noise beyond counting noise."""
    log_likelihood: float

    def ratio(self) -> NDArray[np.float64]:
        """The path on the natural scale (``exp(mean)``)."""
        out: NDArray[np.float64] = np.exp(self.mean)
        return out

    def interval(self, level: float = 0.9) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Pointwise credible band on the natural scale."""
        z = float(norm.ppf(0.5 + level / 2))
        return np.exp(self.mean - z * self.sd), np.exp(self.mean + z * self.sd)

    def change(self, t0: float, t1: float) -> tuple[float, float]:
        """Ratio of the level at ``t1`` to the level at ``t0`` and an approximate SD
        of its log (ignoring smoothing covariance, so conservative)."""
        i0, i1 = int(np.argmin(np.abs(self.t - t0))), int(np.argmin(np.abs(self.t - t1)))
        return math.exp(self.mean[i1] - self.mean[i0]), math.hypot(self.sd[i0], self.sd[i1])


def _filter_smooth(
    y: NDArray[np.float64], obs_var: NDArray[np.float64], dt: NDArray[np.float64], q: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Random walk ``x_t = x_{t-1} + N(0, q*dt)``, ``y_t = x_t + N(0, obs_var_t)``.

    NaN observations are skipped (predict only). Returns smoothed means, variances
    and the log likelihood. The first state has a diffuse prior.
    """
    n = y.size
    m_pred = np.zeros(n)
    p_pred = np.zeros(n)
    m_filt = np.zeros(n)
    p_filt = np.zeros(n)
    ll = 0.0
    m, p = 0.0, 1e7
    first = True
    for i in range(n):
        if i > 0:
            p = p + q * dt[i]
        m_pred[i], p_pred[i] = m, p
        if not np.isnan(y[i]):
            s = p + obs_var[i]
            k = p / s
            resid = y[i] - m
            if not first:
                ll += -0.5 * (math.log(2 * math.pi * s) + resid * resid / s)
            first = False
            m, p = m + k * resid, (1 - k) * p
        m_filt[i], p_filt[i] = m, p
    ms, ps = m_filt.copy(), p_filt.copy()
    for i in range(n - 2, -1, -1):
        g = p_filt[i] / p_pred[i + 1]
        ms[i] = m_filt[i] + g * (ms[i + 1] - m_pred[i + 1])
        ps[i] = p_filt[i] + g * g * (ps[i + 1] - p_pred[i + 1])
    return ms, ps, ll


def _fit(
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    count_var: NDArray[np.float64],
    step_sd: float | None,
    overdispersion_sd: float | None,
) -> StatePath:
    dt = np.diff(t, prepend=t[0])
    if np.any(dt[1:] <= 0):
        raise ValueError("t must be strictly increasing")
    ok = ~np.isnan(y)
    if ok.sum() < 3:
        raise ValueError("need at least 3 periods with observations")
    scale = float(np.median(dt[1:])) if t.size > 1 else 1.0

    def unpack(theta: NDArray[np.float64]) -> tuple[float, float]:
        rest = iter(theta)
        q = (step_sd if step_sd is not None else math.exp(next(rest))) ** 2
        od = (overdispersion_sd if overdispersion_sd is not None else math.exp(next(rest))) ** 2
        return q, od

    def nll(theta: NDArray[np.float64]) -> float:
        q, od = unpack(theta)
        return -_filter_smooth(y, count_var + od, dt, q)[2]

    free = [step_sd is None, overdispersion_sd is None]
    spread = float(np.nanstd(np.diff(y[ok]))) or 0.1
    start = [math.log(spread / math.sqrt(2 * scale))] * free[0] + [math.log(spread / 2)] * free[1]
    theta = (
        minimize(
            nll,
            np.array(start),
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000},
        ).x
        if any(free)
        else np.array([])
    )
    q, od = unpack(theta)
    mean, var, ll = _filter_smooth(y, count_var + od, dt, q)
    return StatePath(t, mean, np.sqrt(var), y, math.sqrt(q), math.sqrt(od), ll)


def _as_array(name: str, x: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(x, float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return arr


def effectiveness_over_time(
    red_losses: ArrayLike,
    blue_losses: ArrayLike,
    *,
    t: ArrayLike | None = None,
    red: ArrayLike | None = None,
    blue: ArrayLike | None = None,
    exponent: float = 1.0,
    step_sd: float | None = None,
    overdispersion_sd: float | None = None,
) -> StatePath:
    """Path of red's per-unit effectiveness relative to blue's, ``log(rho/beta)``.

    :param red_losses, blue_losses: Losses each side suffered in each period.
    :param t: Period times (default 0, 1, 2, ...).
    :param red, blue: Strengths at the start of each period. Required unless
        ``exponent == 1`` (linear law), where they cancel.
    :param exponent: Law exponent ``e = p - q + 1``.
    :param step_sd, overdispersion_sd: Fix instead of estimating.

    Periods where either side lost nothing get a continuity correction of 0.5,
    which the counting noise makes uninformative.
    """
    lr, lb = _as_array("red_losses", red_losses), _as_array("blue_losses", blue_losses)
    if lr.shape != lb.shape:
        raise ValueError("red_losses and blue_losses must have the same length")
    if np.any(lr < 0) or np.any(lb < 0):
        raise ValueError("losses must be non-negative")
    times = np.arange(lr.size, dtype=float) if t is None else _as_array("t", t)
    if times.shape != lr.shape:
        raise ValueError("t must match the losses")
    lr_c, lb_c = lr + 0.5, lb + 0.5
    # log(rho/beta) = (p - q) * log(B/R) - log(lossR/lossB),  p - q = exponent - 1
    y = np.log(lb_c) - np.log(lr_c)
    if exponent != 1.0:
        if red is None or blue is None:
            raise ValueError("strengths are needed unless exponent == 1")
        r, b = _as_array("red", red), _as_array("blue", blue)
        if r.shape != lr.shape or b.shape != lr.shape or np.any(r <= 0) or np.any(b <= 0):
            raise ValueError("strengths must be positive and match the losses")
        y = y + (exponent - 1) * np.log(b / r)
    count_var = 1 / lr_c + 1 / lb_c
    return _fit(times, y, count_var, step_sd, overdispersion_sd)


def intensity_over_time(
    losses: ArrayLike,
    *,
    t: ArrayLike | None = None,
    step_sd: float | None = None,
    overdispersion_sd: float | None = None,
) -> StatePath:
    """Path of the underlying log loss rate behind a noisy per-period loss series.

    ``exp(path.mean)`` is the smoothed losses per period. Negative values (e.g.
    downward corrections in a cumulative source) are treated as missing.
    """
    x = _as_array("losses", losses)
    times = np.arange(x.size, dtype=float) if t is None else _as_array("t", t)
    if times.shape != x.shape:
        raise ValueError("t must match the losses")
    y = np.where(x >= 0, np.log(np.clip(x, 0, None) + 0.5), np.nan)
    count_var = 1 / (np.clip(x, 0, None) + 0.5)
    return _fit(times, y, count_var, step_sd, overdispersion_sd)
