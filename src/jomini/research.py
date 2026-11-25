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

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from jomini.cdb90 import ADVANTAGE_FACTORS, Advantages, HistoricalBattle

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


_POSTURE_DUMMIES = ("prepared", "fortified", "delay", "withdrawal")  # baseline: hasty


def _control_names(adjust_for: str | Sequence[str] | None) -> tuple[str, ...]:
    if adjust_for is None:
        return ()
    names = [adjust_for] if isinstance(adjust_for, str) else list(adjust_for)
    out: list[str] = []
    for name in names:
        if name == "quality":
            out.extend(ADVANTAGE_FACTORS)
        elif name == "all":
            out.extend((*ADVANTAGE_FACTORS, "posture"))
        elif name in ADVANTAGE_FACTORS or name == "posture":
            out.append(name)
        else:
            raise ValueError(
                f"unknown control {name!r}; use 'quality', 'all', 'posture' or one of "
                + ", ".join(ADVANTAGE_FACTORS)
            )
    return tuple(dict.fromkeys(out))


def _control_row(battle: HistoricalBattle, names: tuple[str, ...]) -> list[float] | None:
    row: list[float] = []
    for name in names:
        if name == "posture":
            if battle.defender_posture is None:
                return None
            row.extend(float(battle.defender_posture == p) for p in _POSTURE_DUMMIES)
        else:
            value = getattr(battle.advantages, name)
            if value is None:
                return None
            row.append(float(value))
    return row


def _design(
    observations: Iterable[Observation], adjust_for: str | Sequence[str] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, list[HistoricalBattle]]:
    """Strength rows (R0, R1, B0, B1), optional control matrix, and the battles kept."""
    names = _control_names(adjust_for)
    rows: list[tuple[Any, ...]] = []
    controls: list[list[float]] = []
    kept: list[HistoricalBattle] = []
    for obs in observations:
        if isinstance(obs, HistoricalBattle):
            if not obs.fittable:
                continue
            if names:
                c = _control_row(obs, names)
                if c is None:
                    continue
                controls.append(c)
            a, d = obs.attacker, obs.defender
            rows.append((a.strength, a.remaining, d.strength, d.remaining))
            kept.append(obs)
        else:
            if names:
                raise TypeError("adjust_for needs HistoricalBattle observations, not tuples")
            rows.append(tuple(obs))
    data = np.asarray(rows, dtype=float).reshape(-1, 4)
    r0, r1, b0, b1 = data.T
    if not (np.all(r0 > r1) and np.all(r1 > 0) and np.all(b0 > b1) and np.all(b1 > 0)):
        raise ValueError("each observation needs start > end > 0 for both sides")
    ctrl = np.asarray(controls, dtype=float).reshape(len(rows), -1) if names else None
    return data, ctrl, kept


def _arrays(observations: Iterable[Observation]) -> NDArray[np.float64]:
    return _design(observations)[0]


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


def _slope(
    data: NDArray[np.float64], exponent: float, controls: NDArray[np.float64] | None = None
) -> float:
    """Partial regression slope of log(rho/beta) on log force ratio, given ``controls``."""
    x = np.log(data[:, 0] / data[:, 2])
    y = log_effectiveness_ratio(data, exponent)
    if controls is not None and controls.shape[1]:
        # Frisch-Waugh: residualise both on the controls (with intercept).
        c = np.column_stack([np.ones(len(x)), controls])
        x = x - c @ np.linalg.lstsq(c, x, rcond=None)[0]
        y = y - c @ np.linalg.lstsq(c, y, rcond=None)[0]
    else:
        x = x - x.mean()
        y = y - y.mean()
    return float(np.dot(x, y) / np.dot(x, x))


def _solve(data: NDArray[np.float64], controls: NDArray[np.float64] | None = None) -> float:
    lo, hi = _BRACKET
    return float(brentq(lambda e: _slope(data, e, controls), lo, hi, xtol=1e-6))


def estimate_exponent(
    observations: Iterable[Observation],
    *,
    adjust_for: str | Sequence[str] | None = None,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> ExponentEstimate:
    """Estimate the Lanchester exponent that best describes ``observations``.

    Unfittable :class:`HistoricalBattle` entries are skipped. The confidence
    interval comes from a percentile bootstrap over battles.

    ``adjust_for`` makes effectiveness independent of force ratio *given* other
    factors, so that a side's numbers aren't credited with (or blamed for) its
    quality: ``"quality"`` (all ten CDB90 advantage ratings), ``"all"`` (ratings
    plus defender posture), ``"posture"``, or a list of rating names. Battles
    missing any requested rating are skipped.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    data, controls, _ = _design(observations, adjust_for)
    n_params = 0 if controls is None else controls.shape[1]
    if len(data) < max(10, n_params + 5):
        raise ValueError(f"need at least {max(10, n_params + 5)} battles, got {len(data)}")
    if np.ptp(np.log(data[:, 0] / data[:, 2])) == 0:
        raise ValueError("force ratios must vary across battles")
    try:
        point = _solve(data, controls)
    except ValueError:
        raise ValueError(
            f"no exponent in {_BRACKET} makes effectiveness independent of force ratio"
        ) from None

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(data), len(data))
        try:
            boots.append(_solve(data[idx], None if controls is None else controls[idx]))
        except (ValueError, np.linalg.LinAlgError):
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


# --------------------------------------------------------------------------- combat power


@dataclass(frozen=True)
class Multiplier:
    """Effect of one unit of advantage on the attacker's relative effectiveness."""

    estimate: float
    ci_low: float
    ci_high: float
    t: float

    @property
    def significant(self) -> bool:
        """Whether the 95% interval excludes 1 (no effect)."""
        return self.ci_low > 1 or self.ci_high < 1


@dataclass(frozen=True)
class CombatPowerModel:
    """Effectiveness explained by quality factors: a data-driven take on Dupuy's QJM.

    Models ``log(rho/beta)`` (attacker vs defender per-soldier effectiveness under
    ``exponent``) as a linear function of CDB90's advantage ratings and the
    defender's posture. ``multipliers[f]`` is how much one point of advantage in
    ``f`` multiplies the attacker's effectiveness; posture multipliers are relative
    to a hasty defence.

    The ratings were assigned after the fact by analysts who knew the outcome, so
    treat the multipliers as descriptive associations, not causal effects.
    """

    exponent: float
    baseline: float
    """Attacker effectiveness ratio with no advantages against a hasty defence."""
    multipliers: dict[str, Multiplier]
    r_squared: float
    residual_sd: float
    """Standard deviation of log effectiveness not explained by the factors."""
    n: int

    @classmethod
    def fit(
        cls,
        battles: Iterable[HistoricalBattle] | None = None,
        *,
        exponent: float | None = None,
        posture: bool = True,
    ) -> CombatPowerModel:
        """Fit by least squares. ``exponent`` defaults to the quality-adjusted estimate."""
        from jomini import cdb90

        pool = list(cdb90.battles() if battles is None else battles)
        adjust: tuple[str, ...] = ("all",) if posture else ("quality",)
        if exponent is None:
            exponent = estimate_exponent(pool, adjust_for=adjust, n_boot=0).exponent
        data, controls, _ = _design(pool, adjust)
        assert controls is not None
        names = list(ADVANTAGE_FACTORS) + (
            [f"posture_{p}" for p in _POSTURE_DUMMIES] if posture else []
        )
        # Drop dummies with no variation in this sample (e.g. no withdrawals).
        keep = [j for j in range(controls.shape[1]) if np.ptp(controls[:, j]) > 0]
        names = [names[j] for j in keep]
        x = np.column_stack([np.ones(len(data)), controls[:, keep]])
        if len(data) <= x.shape[1] + 5:
            raise ValueError(f"need more than {x.shape[1] + 5} battles, got {len(data)}")
        y = log_effectiveness_ratio(data, exponent)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ coef
        dof = len(y) - x.shape[1]
        sigma2 = float(resid @ resid) / dof
        se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(x.T @ x)))
        r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
        mult = {
            name: Multiplier(
                estimate=float(np.exp(c)),
                ci_low=float(np.exp(c - 1.96 * s)),
                ci_high=float(np.exp(c + 1.96 * s)),
                t=float(c / s),
            )
            for name, c, s in zip(names, coef[1:], se[1:], strict=True)
        }
        return cls(
            exponent=float(exponent),
            baseline=float(np.exp(coef[0])),
            multipliers=mult,
            r_squared=float(r2),
            residual_sd=float(np.sqrt(sigma2)),
            n=len(y),
        )

    def predict(
        self,
        advantages: Advantages | dict[str, int | None] | None = None,
        defender_posture: str | None = None,
    ) -> float:
        """Expected attacker/defender effectiveness ratio for the given ratings.

        Missing or ``None`` ratings count as 0 (no advantage).
        """
        if isinstance(advantages, Advantages):
            advantages = advantages.as_dict()
        log_ratio = math.log(self.baseline)
        for name, value in (advantages or {}).items():
            if name not in ADVANTAGE_FACTORS:
                raise ValueError(f"unknown factor {name!r}")
            if value and name in self.multipliers:
                log_ratio += value * math.log(self.multipliers[name].estimate)
        if defender_posture not in (None, "hasty"):
            key = f"posture_{defender_posture}"
            if defender_posture not in _POSTURE_DUMMIES:
                raise ValueError(f"unknown posture {defender_posture!r}")
            if key in self.multipliers:
                log_ratio += math.log(self.multipliers[key].estimate)
        return math.exp(log_ratio)

    def predict_battle(self, battle: HistoricalBattle) -> float:
        return self.predict(battle.advantages, battle.defender_posture)

    def __str__(self) -> str:
        lines = [
            f"Combat power model (e = {self.exponent:.2f}, n = {self.n}, "
            f"R² = {self.r_squared:.2f}); attacker effectiveness multipliers:"
        ]
        ranked = sorted(self.multipliers.items(), key=lambda kv: -abs(kv[1].t))
        for name, m in ranked:
            mark = "*" if m.significant else " "
            ci = f"95% CI {m.ci_low:.2f} to {m.ci_high:.2f}"
            lines.append(f"  {mark} {name:22s} x{m.estimate:5.2f}  ({ci})")
        return "\n".join(lines)


@cache
def combat_power_model() -> CombatPowerModel:
    """The combat power model fitted to all of CDB90 (cached)."""
    return CombatPowerModel.fit()
