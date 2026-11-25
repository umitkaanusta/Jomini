"""Which Lanchester law do real battles follow?

A single battle can't tell the laws apart: any of them can be fitted exactly to
one pair of losses. Across many battles they can be told apart. Bracken's general law
conserves a quantity along every trajectory::

    rho * (R0**e - R1**e) / e  ==  beta * (B0**e - B1**e) / e,   e = p - q + 1

(``e -> 0`` gives the logarithmic form ``rho*ln(R0/R1) == beta*ln(B0/B1)``), so each
battle's start and end strengths give its relative effectiveness ``rho / beta`` for
any law exponent ``e``. Fighting quality shouldn't depend on how many troops a side
brings, so the right ``e`` is the one at which ``log(rho/beta)`` is
**uncorrelated with the force ratio** ``log(R0/B0)`` across battles (Helmbold's
criterion). ``e = 2`` is the square law, ``1`` the linear law and ``0`` the
logarithmic law.

Caveat: measurement error in the recorded strengths enters both sides of the
regression, which can bias the estimate. Treat results as descriptive.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from jomini.cdb90 import HistoricalBattle

Observation = HistoricalBattle | tuple[float, float, float, float]
"""A fittable :class:`HistoricalBattle`, or ``(red_start, red_end, blue_start, blue_end)``."""

_LAWS = {2.0: "square", 1.0: "linear", 0.0: "logarithmic"}
_BRACKET = (-3.0, 5.0)


@dataclass(frozen=True)
class ExponentEstimate:
    """Estimated law exponent ``e = p - q + 1`` with a bootstrap confidence interval."""

    exponent: float
    ci_low: float
    ci_high: float
    confidence: float
    n: int

    @property
    def nearest_law(self) -> str:
        """The classic law closest to the estimate: square, linear or logarithmic."""
        return _LAWS[min(_LAWS, key=lambda e: abs(e - self.exponent))]

    def excludes(self, law: str) -> bool:
        """Whether the confidence interval rules out ``law`` (square/linear/logarithmic)."""
        e = {v: k for k, v in _LAWS.items()}[law]
        return not self.ci_low <= e <= self.ci_high

    def __str__(self) -> str:
        pct = round(self.confidence * 100)
        return (
            f"e = {self.exponent:.2f} ({pct}% CI {self.ci_low:.2f} to {self.ci_high:.2f}, "
            f"n = {self.n}); nearest: {self.nearest_law} law"
        )


def _arrays(observations: Iterable[Observation]) -> NDArray[np.float64]:
    rows: list[tuple[Any, ...]] = []
    for obs in observations:
        if isinstance(obs, HistoricalBattle):
            if not obs.fittable:
                continue
            a, d = obs.attacker, obs.defender
            rows.append((a.strength, a.remaining, d.strength, d.remaining))
        else:
            rows.append(tuple(obs))
    data = np.asarray(rows, dtype=float).reshape(-1, 4)
    r0, r1, b0, b1 = data.T
    if not (np.all(r0 > r1) and np.all(r1 > 0) and np.all(b0 > b1) and np.all(b1 > 0)):
        raise ValueError("each observation needs start > end > 0 for both sides")
    return data


def log_effectiveness_ratio(data: NDArray[np.float64], exponent: float) -> NDArray[np.float64]:
    """``log(rho / beta)`` per row of ``data`` (columns: R0, R1, B0, B1) under ``exponent``."""
    r0, r1, b0, b1 = data.T
    if abs(exponent) < 1e-9:
        num, den = np.log(b0 / b1), np.log(r0 / r1)
    else:
        # Scale by the starting strengths before powering to avoid overflow for large e.
        num = b0**exponent * (1 - (b1 / b0) ** exponent) / exponent
        den = r0**exponent * (1 - (r1 / r0) ** exponent) / exponent
    out: NDArray[np.float64] = np.log(num) - np.log(den)
    return out


def _slope(data: NDArray[np.float64], exponent: float) -> float:
    x = np.log(data[:, 0] / data[:, 2])
    y = log_effectiveness_ratio(data, exponent)
    x = x - x.mean()
    return float(np.dot(x, y - y.mean()) / np.dot(x, x))


def _solve(data: NDArray[np.float64]) -> float:
    lo, hi = _BRACKET
    return float(brentq(lambda e: _slope(data, e), lo, hi, xtol=1e-6))


def estimate_exponent(
    observations: Iterable[Observation],
    *,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> ExponentEstimate:
    """Estimate the Lanchester exponent that best describes ``observations``.

    Unfittable :class:`HistoricalBattle` entries are skipped. The confidence
    interval comes from a percentile bootstrap over battles.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    data = _arrays(observations)
    if len(data) < 10:
        raise ValueError(f"need at least 10 battles, got {len(data)}")
    if np.ptp(np.log(data[:, 0] / data[:, 2])) == 0:
        raise ValueError("force ratios must vary across battles")
    try:
        point = _solve(data)
    except ValueError:
        raise ValueError(
            f"no exponent in {_BRACKET} makes effectiveness independent of force ratio"
        ) from None

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = data[rng.integers(0, len(data), len(data))]
        try:
            boots.append(_solve(sample))
        except ValueError:
            continue  # resample without a root inside the bracket
    if not boots:
        return ExponentEstimate(point, np.nan, np.nan, confidence, len(data))
    tail = (1 - confidence) / 2 * 100
    lo, hi = np.percentile(boots, [tail, 100 - tail])
    return ExponentEstimate(point, float(lo), float(hi), confidence, len(data))


def relative_effectiveness(observation: Observation, exponent: float = 1.0) -> float:
    """``rho / beta``: how many times as effective red (the attacker) was as blue."""
    return float(np.exp(log_effectiveness_ratio(_arrays([observation]), exponent))[0])


def effectiveness_table(
    battles: Sequence[HistoricalBattle] | None = None, exponent: float = 1.0
) -> Any:
    """One row per fittable battle with its relative effectiveness (needs ``jomini[pandas]``).

    ``attacker_effectiveness`` is ``rho / beta`` under ``exponent``: above 1, each
    attacking soldier was more effective than each defender.
    """
    import pandas as pd

    from jomini import cdb90

    pool = [b for b in (cdb90.battles() if battles is None else battles) if b.fittable]
    ratios = np.exp(log_effectiveness_ratio(_arrays(pool), exponent))
    return pd.DataFrame(
        {
            "name": [b.name for b in pool],
            "year": [b.year for b in pool],
            "war": [b.war for b in pool],
            "attacker": [", ".join(b.attacker.actors) for b in pool],
            "defender": [", ".join(b.defender.actors) for b in pool],
            "force_ratio": [b.attacker.strength / b.defender.strength for b in pool],  # type: ignore[operator]
            "attacker_effectiveness": ratios,
            "winner": [b.winner for b in pool],
        },
        index=pd.Index([b.id for b in pool], name="id"),
    )
