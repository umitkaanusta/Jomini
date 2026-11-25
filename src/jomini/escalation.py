"""Self-exciting (Hawkes) models of conflict escalation for daily event or death counts.

Violence clusters: an attack makes more attacks likelier for a while (retaliation,
offensives unfolding, contagion). A discrete-time Hawkes process captures this::

    rate_t = baseline_t + alpha * S_t,    S_t = theta * S_{t-1} + (1 - theta) * y_{t-1}

``alpha`` is the **branching ratio**, the expected number of follow-on events each
event triggers (equivalently the long-run share of events that are triggered rather
than spontaneous; below 1 for a stable process). ``theta`` sets how fast excitation
decays (half-life ``log(0.5) / log(theta)`` days). Counts follow a negative binomial
around the rate, since conflict data are overdispersed.

A war whose intensity drifts for exogenous reasons (mobilisation, offensives, lulls)
can masquerade as self-excitation: with a constant baseline, slow persistence pushes
the branching ratio towards 1. The default ``baseline="spline"`` lets the background
rate follow a smooth curve (a knot every ``knot_days``), so ``alpha`` measures
short-term clustering on top of slow trends. Compare fits; ``alpha`` near 1 signals
unmodelled nonstationarity rather than runaway contagion.

Calibration (simulations, true branching ratio 0.5, 400-1500 days): the spline fit
has a median of 0.41 with 90-day knots and 0.44-0.46 with 180-day knots, so it
understates self-excitation by roughly 10-20%. The constant fit is unbiased only when
the background really is constant. With a drifting background and no excitation at
all, the constant fit reported 0.98 (p < 1e-200) while the spline fit reported 0.00.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline
from scipy.optimize import OptimizeResult, minimize
from scipy.signal import lfilter
from scipy.special import expit, gammaln
from scipy.stats import chi2

Baseline = Literal["spline", "trend", "constant"]


@dataclass(frozen=True)
class HawkesFit:
    baseline_rate: float
    """Background events per day (at the series midpoint for a trend baseline)."""
    trend_per_year: float
    """Multiplicative change in the background rate per year (1.0 = constant)."""
    branching_ratio: float
    decay: float
    dispersion: float
    """Negative binomial size: variance = mean + mean**2 / dispersion."""
    log_likelihood: float
    null_log_likelihood: float
    """Same model with no self-excitation (alpha = 0)."""
    n: int
    params: int
    baseline_path: NDArray[np.float64] = field(repr=False, default_factory=lambda: np.zeros(0))
    """Background rate per day over the whole series (smooth part only)."""

    @property
    def half_life_days(self) -> float:
        return math.log(0.5) / math.log(self.decay) if 0 < self.decay < 1 else math.inf

    @property
    def lr_statistic(self) -> float:
        return 2 * (self.log_likelihood - self.null_log_likelihood)

    @property
    def p_value(self) -> float:
        """Likelihood-ratio test of self-excitation (alpha and theta vs none).

        The null sits on the boundary (alpha = 0) and theta is unidentified there, so
        a chi-square with 2 degrees of freedom is conservative.
        """
        return float(chi2.sf(max(self.lr_statistic, 0.0), 2))

    @property
    def aic(self) -> float:
        return 2 * self.params - 2 * self.log_likelihood

    def __str__(self) -> str:
        trend = f", trend x{self.trend_per_year:.2f}/yr" if self.trend_per_year != 1 else ""
        return (
            f"branching ratio {self.branching_ratio:.2f}, half-life "
            f"{self.half_life_days:.1f} days, baseline {self.baseline_rate:.2f}/day{trend}; "
            f"LR test p = {self.p_value:.2g} (n = {self.n} days)"
        )


def _nb_loglik(y: NDArray[np.float64], mu: NDArray[np.float64], r: float) -> float:
    mu = np.maximum(mu, 1e-9)
    return float(
        np.sum(
            gammaln(y + r)
            - gammaln(r)
            - gammaln(y + 1)
            + r * np.log(r / (r + mu))
            + y * np.log(mu / (r + mu))
        )
    )


def _excitation(y: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
    """S_t = theta*S_{t-1} + (1-theta)*y_{t-1}, vectorised with a linear filter."""
    out: NDArray[np.float64] = lfilter([0.0, 1 - theta], [1.0, -theta], y)
    return out


def _spline_basis(n: int, knot_days: int) -> NDArray[np.float64]:
    """Cubic B-spline basis over days 0..n-1 with interior knots every ``knot_days``."""
    k = 3
    inner = np.arange(0, n, knot_days, dtype=float)
    if inner[-1] < n - 1:
        inner = np.append(inner, n - 1)
    knots = np.r_[[inner[0]] * k, inner, [inner[-1]] * k]
    x = np.arange(n, dtype=float)
    basis: NDArray[np.float64] = BSpline.design_matrix(x, knots, k).toarray()
    return basis


def fit_hawkes(
    counts: ArrayLike,
    *,
    baseline: Baseline = "spline",
    knot_days: int = 90,
    burn_in: int = 30,
) -> HawkesFit:
    """Fit a discrete-time Hawkes model to daily counts by maximum likelihood.

    :param counts: Non-negative daily counts (events, or deaths rounded).
    :param baseline: ``"spline"`` (default: background rate follows a smooth curve
        with a knot every ``knot_days``, so slow escalation isn't mistaken for
        contagion), ``"trend"`` (log-linear drift) or ``"constant"``.
    :param burn_in: Days used only to build up the excitation state, not scored.
    """
    y = np.asarray(counts, float)
    if y.ndim != 1 or y.size < burn_in + 30:
        raise ValueError(f"need a 1-D series of at least {burn_in + 30} days")
    if np.any(y < 0) or not np.all(np.isfinite(y)):
        raise ValueError("counts must be non-negative and finite")
    if y[burn_in:].sum() == 0:
        raise ValueError("no events after the burn-in period")
    if baseline not in ("spline", "trend", "constant"):
        raise ValueError("baseline must be 'spline', 'trend' or 'constant'")
    if knot_days < 14:
        raise ValueError("knot_days must be at least 14")
    y = np.round(y)
    n = y.size
    if baseline == "spline":
        design = _spline_basis(n, knot_days)
    elif baseline == "trend":
        design = np.column_stack([np.ones(n), (np.arange(n) - n / 2) / 365.25])
    else:
        design = np.ones((n, 1))
    nb = design.shape[1]
    scored = slice(burn_in, None)
    ys = y[scored]
    # Light smoothness penalty on neighbouring spline coefficients (second differences).
    penalty = np.diff(np.eye(nb), 2, axis=0) if baseline == "spline" and nb > 2 else None

    def split(
        x: NDArray[np.float64], excite: bool
    ) -> tuple[NDArray[np.float64], float, float, float]:
        beta = x[:nb]
        alpha = float(expit(x[nb])) if excite else 0.0
        theta = float(expit(x[nb + 1])) if excite else 0.5
        r = math.exp(x[-1])
        return beta, alpha, theta, r

    def nll(x: NDArray[np.float64], excite: bool) -> float:
        beta, alpha, theta, r = split(x, excite)
        mu = np.exp(np.clip(design @ beta, -30, 30))
        if excite:
            mu = mu + alpha * _excitation(y, theta)
        value = -_nb_loglik(ys, mu[scored], r)
        if penalty is not None:
            value += 0.5 * float(np.sum((penalty @ beta) ** 2))
        return value

    mean0 = max(float(ys.mean()), 1e-3)
    base0 = (
        np.full(nb, math.log(mean0))
        if baseline == "spline"
        else np.r_[math.log(mean0), np.zeros(nb - 1)]
    )

    def run(excite: bool, start_excite: tuple[float, float]) -> OptimizeResult:
        x0 = np.r_[base0, list(start_excite) if excite else [], 0.0]
        bounds = [(-30.0, 30.0)] * nb + [(-20.0, 20.0)] * (2 if excite else 0) + [(-10.0, 15.0)]
        return minimize(nll, x0, args=(excite,), method="L-BFGS-B", bounds=bounds)

    null = run(False, (0, 0))
    best = min((run(True, s) for s in ((-1.0, 1.5), (0.5, 0.0), (-3.0, 3.0))), key=lambda r: r.fun)
    beta, alpha, theta, r = split(best.x, True)
    rate_path = np.exp(np.clip(design @ beta, -30, 30))
    if baseline == "trend":
        trend = math.exp(beta[1])
        base_rate = math.exp(beta[0])
    else:
        trend = 1.0
        base_rate = float(np.median(rate_path))

    # Report likelihoods without the smoothness penalty.
    def ll(x: NDArray[np.float64], excite: bool) -> float:
        b, a, th, rr = split(x, excite)
        mu = np.exp(np.clip(design @ b, -30, 30)) + (a * _excitation(y, th) if excite else 0.0)
        return _nb_loglik(ys, mu[scored], rr)

    ll_fit, ll_null = ll(best.x, True), ll(null.x, False)
    return HawkesFit(
        baseline_rate=base_rate,
        trend_per_year=trend,
        branching_ratio=alpha,
        decay=theta,
        dispersion=r,
        log_likelihood=max(ll_fit, ll_null),
        null_log_likelihood=ll_null,
        n=int(n - burn_in),
        params=int(best.x.size),
        baseline_path=rate_path,
    )


def simulate_hawkes(
    days: int,
    *,
    baseline_rate: float,
    branching_ratio: float,
    decay: float,
    dispersion: float = math.inf,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate daily counts from the discrete-time Hawkes model (for testing and
    what-if exploration). ``dispersion=inf`` gives Poisson counts."""
    if not 0 <= branching_ratio < 1:
        raise ValueError("branching_ratio must be in [0, 1)")
    if not 0 <= decay < 1:
        raise ValueError("decay must be in [0, 1)")
    rng = np.random.default_rng(seed)
    y = np.zeros(days)
    s = 0.0
    for t in range(days):
        if t > 0:
            s = decay * s + (1 - decay) * y[t - 1]
        mu = baseline_rate + branching_ratio * s
        if math.isinf(dispersion):
            y[t] = rng.poisson(mu)
        else:
            y[t] = rng.negative_binomial(dispersion, dispersion / (dispersion + mu))
    return y
