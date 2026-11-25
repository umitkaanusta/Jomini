"""Breakpoints and Monte Carlo: who wins, with what probability.

Real battles end when one side *breaks*, long before it is destroyed: in CDB90 the
loser lost a median of 15% of its strength. :class:`BreakpointModel` describes the
fraction of its strength a side will lose before breaking as a random variable,
fitted to history. :func:`monte_carlo` combines it with any Lanchester model:
it samples breakpoints (and optionally per-soldier effectiveness), runs the
attrition model, and reports how often each side wins.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import norm

from jomini.battle import Battle
from jomini.cdb90 import HistoricalBattle
from jomini.models import LanchesterModel

Role = Literal["attacker", "defender"]
_REF_YEAR = 1900.0


# --------------------------------------------------------------------------- breakpoints


@dataclass(frozen=True)
class BreakpointModel:
    """Log-normal model of the fraction of its strength a side loses before breaking.

    ``log(breakpoint) ~ Normal(mu, sigma)`` with
    ``mu = intercept + year_slope * (year - 1900) / 100 + defender_shift * is_defender``.

    Fitted by maximum likelihood with censoring: a losing side's casualty fraction
    *is* its breakpoint, while the winner's only tells us its breakpoint was higher.
    """

    intercept: float
    year_slope: float
    """Change in log-breakpoint per century."""
    defender_shift: float
    """Defenders' log-breakpoint relative to attackers'."""
    sigma: float
    n: int = 0
    log_likelihood: float = field(default=math.nan, compare=False)

    def mu(self, year: float, role: Role) -> float:
        shift = self.defender_shift if role == "defender" else 0.0
        return self.intercept + self.year_slope * (year - _REF_YEAR) / 100 + shift

    def median(self, year: float, role: Role) -> float:
        """Median fraction of its strength a side loses before breaking."""
        return min(math.exp(self.mu(year, role)), 1.0)

    def quantile(self, q: float, year: float, role: Role) -> float:
        return min(math.exp(self.mu(year, role) + self.sigma * float(norm.ppf(q))), 1.0)

    def sample(
        self, rng: np.random.Generator, size: int, year: float, role: Role
    ) -> NDArray[np.float64]:
        """Random breakpoints, capped at 1 (fight to the end)."""
        draws = np.exp(self.mu(year, role) + self.sigma * rng.standard_normal(size))
        out: NDArray[np.float64] = np.minimum(draws, 1.0)
        return out

    @classmethod
    def fit(cls, battles: Iterable[HistoricalBattle] | None = None) -> BreakpointModel:
        """Fit to historical battles (all of CDB90 by default)."""
        from jomini import cdb90

        rows = []  # (log_fraction, year, is_defender, broke)
        for b in cdb90.battles() if battles is None else battles:
            for role, side in (("attacker", b.attacker), ("defender", b.defender)):
                frac = side.loss_fraction
                if frac is None or not 0 < frac < 1:
                    continue
                broke = b.winner is not None and b.winner != "draw" and b.winner != role
                rows.append((math.log(frac), b.year, role == "defender", broke))
        if len(rows) < 20:
            raise ValueError(f"need at least 20 side observations, got {len(rows)}")
        data = np.array(rows, dtype=float)
        y, year, is_def, broke = data.T
        t = (year - _REF_YEAR) / 100
        event = broke.astype(bool)

        def nll(theta: NDArray[np.float64]) -> float:
            a, b, c, log_s = theta
            s = math.exp(log_s)
            z = (y - (a + b * t + c * is_def)) / s
            ll = np.where(event, norm.logpdf(z) - log_s, norm.logsf(z))
            return -float(ll.sum())

        start = np.array([float(y[event].mean()), 0.0, 0.0, math.log(float(y.std()) + 1e-3)])
        res = minimize(nll, start, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6})
        if not res.success:
            raise RuntimeError(f"breakpoint fit did not converge: {res.message}")
        a, b, c, log_s = res.x
        return cls(float(a), float(b), float(c), math.exp(log_s), len(rows), -float(res.fun))


@cache
def default_breakpoints() -> BreakpointModel:
    """Breakpoint model fitted to all of CDB90 (cached)."""
    return BreakpointModel.fit()


# --------------------------------------------------------------------------- Monte Carlo


@dataclass(frozen=True)
class MonteCarloResult:
    """Outcomes of many randomised runs of the same battle.

    ``winners`` holds ``"red"``, ``"blue"`` or ``""`` (undecided at ``t_max``) per run;
    the other arrays hold that run's duration and casualties.
    """

    winners: NDArray[np.str_]
    durations: NDArray[np.float64]
    red_casualties: NDArray[np.float64]
    blue_casualties: NDArray[np.float64]

    @property
    def n(self) -> int:
        return int(self.winners.size)

    @property
    def p_red(self) -> float:
        return float(np.mean(self.winners == "red"))

    @property
    def p_blue(self) -> float:
        return float(np.mean(self.winners == "blue"))

    @property
    def p_undecided(self) -> float:
        return float(np.mean(self.winners == ""))

    def quantiles(
        self, values: NDArray[np.float64], qs: tuple[float, ...] = (0.1, 0.5, 0.9)
    ) -> tuple[float, ...]:
        return tuple(float(v) for v in np.quantile(values, qs))

    def __str__(self) -> str:
        def q(v: NDArray[np.float64]) -> str:
            lo, mid, hi = self.quantiles(v)
            return f"{mid:,.0f} (80% range {lo:,.0f} to {hi:,.0f})"

        out = (
            f"{self.n:,} runs: red wins {self.p_red:.1%}, blue wins {self.p_blue:.1%}"
            + (f", undecided {self.p_undecided:.1%}" if self.p_undecided else "")
            + f"\nRed casualties:  {q(self.red_casualties)}"
            + f"\nBlue casualties: {q(self.blue_casualties)}"
        )
        lo, mid, hi = self.quantiles(self.durations)
        return out + f"\nDuration: {mid:.3g} (80% range {lo:.3g} to {hi:.3g})"


def monte_carlo(
    model: LanchesterModel,
    battle: Battle,
    *,
    n: int = 2000,
    year: float = 1900,
    red_role: Role = "attacker",
    breakpoints: BreakpointModel | tuple[float, float] | None = None,
    effectiveness_sd: float = 0.0,
    t_max: float | None = None,
    min_strength: float = 1.0,
    seed: int | None = None,
) -> MonteCarloResult:
    """Fight ``battle`` ``n`` times with random breakpoints and effectiveness.

    :param breakpoints: A :class:`BreakpointModel` (default: fitted to CDB90 and
        evaluated at ``year``), or fixed ``(red, blue)`` loss fractions.
    :param red_role: Whether red is the attacker or the defender (defenders tend to
        hold longer).
    :param effectiveness_sd: Standard deviation of random noise on
        ``log(rho/beta)``, i.e. uncertainty about relative quality. 0 keeps the
        model's coefficients fixed. ``combat_power_model().residual_sd`` (~0.9) is
        the spread history shows after accounting for rated factors.
    :param min_strength: A side below this strength counts as destroyed.
    """
    if not isinstance(model, LanchesterModel):
        raise TypeError("model must be a Lanchester model")
    if not isinstance(battle, Battle):
        raise TypeError("battle must be a jomini.Battle")
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if effectiveness_sd < 0 or not math.isfinite(effectiveness_sd):
        raise ValueError("effectiveness_sd must be a non-negative number")
    if t_max is not None and not (math.isfinite(t_max) and t_max > 0):
        raise ValueError("t_max must be positive")
    if not 0 <= min_strength < min(battle.red, battle.blue):
        raise ValueError("min_strength must be >= 0 and below both sides' initial strength")
    if red_role not in ("attacker", "defender"):
        raise ValueError("red_role must be 'attacker' or 'defender'")

    rng = np.random.default_rng(seed)
    r0, b0 = float(battle.red), float(battle.blue)
    blue_role: Role = "defender" if red_role == "attacker" else "attacker"
    if breakpoints is None:
        breakpoints = default_breakpoints()
    if isinstance(breakpoints, BreakpointModel):
        thr_r = breakpoints.sample(rng, n, year, red_role)
        thr_b = breakpoints.sample(rng, n, year, blue_role)
    else:
        if len(breakpoints) != 2 or not all(0 < float(x) <= 1 for x in breakpoints):
            raise ValueError("fixed breakpoints must be a (red, blue) pair in (0, 1]")
        thr_r = np.full(n, float(breakpoints[0]))
        thr_b = np.full(n, float(breakpoints[1]))
    # A side can never lose more than down to min_strength.
    thr_r = np.minimum(thr_r, 1 - min_strength / r0)
    thr_b = np.minimum(thr_b, 1 - min_strength / b0)

    z = rng.standard_normal(n) * effectiveness_sd
    rho = model.rho * np.exp(z / 2)
    beta = model.beta * np.exp(-z / 2)
    p, q = model.exponents

    def rhs(_t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        red = np.maximum(y[:n], 0.0)
        blue = np.maximum(y[n:], 0.0)
        return np.concatenate([-beta * blue**p * red**q, -rho * red**p * blue**q])

    red_loss, blue_loss = model.casualty_rates(r0, b0)
    timescale = min(r0 / red_loss, b0 / blue_loss)
    horizon = t_max if t_max is not None else 4 * timescale
    y0 = np.concatenate([np.full(n, r0), np.full(n, b0)])

    for _ in range(60):
        sol = solve_ivp(
            rhs, (0.0, horizon), y0, method="RK45", dense_output=True, rtol=1e-6, atol=1e-6
        )
        if not sol.success:
            raise RuntimeError(f"integration failed: {sol.message}")
        grid = np.linspace(0.0, horizon, 1201)
        states = np.clip(sol.sol(grid), 0.0, None)  # (2n, len(grid))
        loss_r = 1 - states[:n] / r0
        loss_b = 1 - states[n:] / b0
        t_r = _crossing(grid, loss_r, thr_r)
        t_b = _crossing(grid, loss_b, thr_b)
        if t_max is not None or np.all(np.isfinite(np.minimum(t_r, t_b))):
            break
        horizon *= 4
    else:
        raise RuntimeError("some runs never reached a breakpoint; pass t_max")

    end = np.minimum(np.minimum(t_r, t_b), horizon)
    winners = np.where(t_r < t_b, "blue", np.where(t_b < t_r, "red", ""))
    winners = np.where(np.isfinite(np.minimum(t_r, t_b)), winners, "")
    # Strength at each run's end time.
    idx = np.arange(n)
    final = sol.sol(end)  # (2n, n): column j is the state at end[j]
    red_end = np.clip(final[idx, idx], 0, r0)
    blue_end = np.clip(final[n + idx, idx], 0, b0)
    return MonteCarloResult(
        winners=winners.astype(str),
        durations=end,
        red_casualties=r0 - red_end,
        blue_casualties=b0 - blue_end,
    )


def _crossing(
    grid: NDArray[np.float64], loss: NDArray[np.float64], threshold: NDArray[np.float64]
) -> NDArray[np.float64]:
    """First time each row of ``loss`` reaches its ``threshold`` (inf if never)."""
    above = loss >= threshold[:, None]
    hit = above.any(axis=1)
    k = np.argmax(above, axis=1)
    out = np.full(loss.shape[0], np.inf)
    rows = np.flatnonzero(hit & (k > 0))
    k1 = k[rows]
    l0, l1 = loss[rows, k1 - 1], loss[rows, k1]
    frac = np.where(l1 > l0, (threshold[rows] - l0) / (l1 - l0), 1.0)
    out[rows] = grid[k1 - 1] + frac * (grid[k1] - grid[k1 - 1])
    out[np.flatnonzero(hit & (k == 0))] = 0.0
    return out
