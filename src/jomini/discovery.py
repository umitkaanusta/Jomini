"""Learn attrition laws from loss time series (daily casualties, equipment losses).

* :func:`fit_power_law` fits Bracken's law separately for each side,
  ``dR/dt = -beta * B**p * R**q`` (and likewise for blue), by linear regression of
  ``log(loss rate)`` on ``log(enemy)`` and ``log(own)`` strength. Each side gets its
  own exponents, so asymmetric laws (e.g. aimed fire one way, area fire the
  other) show up directly.
* :func:`sindy` is sparse identification of nonlinear dynamics (SINDy): it
  regresses the loss rates on a library of polynomial terms and prunes small
  coefficients (sequentially thresholded least squares), so the law is
  discovered rather than assumed.

Both work on any strength series, such as daily equipment-loss counts from the
Russia-Ukraine war. Rates come from finite differences, so smooth noisy data first.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _series(t: ArrayLike, red: ArrayLike, blue: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    ts, rs, bs = (np.asarray(x, float) for x in (t, red, blue))
    if not (ts.ndim == rs.ndim == bs.ndim == 1 and ts.size == rs.size == bs.size):
        raise ValueError("t, red and blue must be 1-D sequences of equal length")
    if ts.size < 5:
        raise ValueError("need at least 5 observations")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("t must be strictly increasing")
    if (
        not (np.all(np.isfinite(rs)) and np.all(np.isfinite(bs)))
        or np.any(rs <= 0)
        or np.any(bs <= 0)
    ):
        raise ValueError("strengths must be positive and finite")
    return ts, rs, bs


def _rates(t: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Loss rate (positive when strength falls) by second-order finite differences."""
    out: NDArray[np.float64] = -np.gradient(x, t)
    return out


@dataclass(frozen=True)
class SideLaw:
    """``own_loss_rate = coefficient * enemy**p * own**q`` with standard errors."""

    coefficient: float
    p: float
    q: float
    p_se: float
    q_se: float
    r_squared: float
    n: int

    @property
    def exponent(self) -> float:
        """``e = p - q + 1`` for this side (2 square, 1 linear, 0 logarithmic)."""
        return self.p - self.q + 1


@dataclass(frozen=True)
class PowerLawFit:
    red: SideLaw
    """Red's losses as a function of blue (enemy) and red (own) strength."""
    blue: SideLaw

    def __str__(self) -> str:
        def fmt(name: str, s: SideLaw) -> str:
            return (
                f"{name} losses = {s.coefficient:.3g} * enemy^{s.p:.2f}(±{s.p_se:.2f}) "
                f"* own^{s.q:.2f}(±{s.q_se:.2f})   R² = {s.r_squared:.2f}, n = {s.n}"
            )

        return fmt("Red ", self.red) + "\n" + fmt("Blue", self.blue)


def _fit_side(
    rate: NDArray[np.float64], enemy: NDArray[np.float64], own: NDArray[np.float64]
) -> SideLaw:
    keep = rate > 0
    if keep.sum() < 4:
        raise ValueError("need at least 4 time steps with positive losses for each side")
    y = np.log(rate[keep])
    x = np.column_stack([np.ones(keep.sum()), np.log(enemy[keep]), np.log(own[keep])])
    if np.linalg.cond(x) > 1e8:
        raise ValueError(
            "own and enemy strengths move in lockstep, so the exponents can't be separated; "
            "pass several trajectories with different starting strengths"
        )
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    dof = max(len(y) - 3, 1)
    cov = float(resid @ resid) / dof * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss if ss > 0 else math.nan
    return SideLaw(
        math.exp(coef[0]),
        float(coef[1]),
        float(coef[2]),
        float(se[1]),
        float(se[2]),
        r2,
        int(keep.sum()),
    )


def fit_power_law(
    trajectories: Sequence[tuple[ArrayLike, ArrayLike, ArrayLike]],
) -> PowerLawFit:
    """Fit Bracken's power law to each side's loss rate from ``(t, red, blue)`` series.

    One trajectory is enough when own and enemy strength fall at different relative
    speeds (square and linear laws). Under the logarithmic law both decay
    exponentially in lockstep, so pass several trajectories with different starting
    strengths; a ``ValueError`` says when the exponents can't be separated.
    """
    if not trajectories:
        raise ValueError("pass at least one (t, red, blue) trajectory")
    rates_r, rates_b, reds, blues = [], [], [], []
    for tr in trajectories:
        ts, rs, bs = _series(*tr)
        rates_r.append(_rates(ts, rs))
        rates_b.append(_rates(ts, bs))
        reds.append(rs)
        blues.append(bs)
    r, b = np.concatenate(reds), np.concatenate(blues)
    return PowerLawFit(
        red=_fit_side(np.concatenate(rates_r), b, r),
        blue=_fit_side(np.concatenate(rates_b), r, b),
    )


@dataclass(frozen=True)
class SindyModel:
    """Discovered ``dR/dt`` and ``dB/dt`` as sparse sums of library terms."""

    terms: tuple[str, ...]
    red: dict[str, float]
    """Non-zero coefficients of ``dR/dt``."""
    blue: dict[str, float]
    scale: float
    """Strengths were divided by this before fitting; coefficients are in scaled units."""

    def equations(self) -> str:
        def eq(name: str, coefs: dict[str, float]) -> str:
            body = " ".join(f"{c:+.4g}·{k}" if k != "1" else f"{c:+.4g}" for k, c in coefs.items())
            return f"d{name}/dt = {body or '0'}"

        return eq("R", self.red) + "\n" + eq("B", self.blue)

    def __str__(self) -> str:
        return self.equations() + f"\n(R, B in units of {self.scale:g})"


def sindy(
    trajectories: Sequence[tuple[ArrayLike, ArrayLike, ArrayLike]],
    *,
    degree: int = 2,
    threshold: float = 0.05,
    max_iter: int = 10,
) -> SindyModel:
    """Discover the attrition law with sequentially thresholded least squares.

    ``trajectories`` is a list of ``(t, red, blue)`` series. **Pass several battles
    with different starting strengths:** along a single trajectory, Lanchester laws
    conserve a quantity (e.g. ``rho*R**2 - beta*B**2`` for the square law), which
    makes library terms linearly dependent so the law cannot be identified.

    The library holds all monomials ``R**i * B**j`` with ``i + j <= degree``.
    Strengths are scaled by the largest initial value so ``threshold`` is comparable
    across datasets; coefficients smaller than ``threshold`` times the largest
    coefficient in that equation are pruned.
    """
    if not trajectories:
        raise ValueError("pass at least one (t, red, blue) trajectory")
    series = [_series(*tr) for tr in trajectories]
    if not 1 <= degree <= 4:
        raise ValueError("degree must be between 1 and 4")
    if not 0 <= threshold < 1:
        raise ValueError("threshold must be in [0, 1)")
    scale = float(max(max(rs[0], bs[0]) for _, rs, bs in series))
    powers = [(i, j) for i, j in itertools.product(range(degree + 1), repeat=2) if i + j <= degree]

    def term(i: int, j: int) -> str:
        parts = [f"R^{i}" if i > 1 else "R" if i else "", f"B^{j}" if j > 1 else "B" if j else ""]
        return "·".join(p for p in parts if p) or "1"

    names = tuple(term(i, j) for i, j in powers)
    rows, dr, db = [], [], []
    for ts, rs, bs in series:
        r, b = rs / scale, bs / scale
        rows.append(np.column_stack([r**i * b**j for i, j in powers]))
        dr.append(np.gradient(r, ts))
        db.append(np.gradient(b, ts))
    theta = np.vstack(rows)
    norms = np.linalg.norm(theta, axis=0)
    norms[norms == 0] = 1
    theta_n = theta / norms

    def stlsq(dx: NDArray[np.float64]) -> NDArray[np.float64]:
        xi = np.linalg.lstsq(theta_n, dx, rcond=None)[0]
        for _ in range(max_iter):
            small = np.abs(xi) < threshold * np.abs(xi).max()
            xi[small] = 0
            big = ~small
            if not big.any():
                break
            xi[big] = np.linalg.lstsq(theta_n[:, big], dx, rcond=None)[0]
        out: NDArray[np.float64] = xi / norms
        return out

    xr, xb = stlsq(np.concatenate(dr)), stlsq(np.concatenate(db))
    return SindyModel(
        terms=names,
        red={n: float(c) for n, c in zip(names, xr, strict=True) if c != 0},
        blue={n: float(c) for n, c in zip(names, xb, strict=True) if c != 0},
        scale=scale,
    )
