"""MCP server that turns Jomini into tools for AI agents ("AI military historian").

Run with ``jomini-mcp`` (needs ``pip install "jomini[mcp]"``). Tools return JSON.
"""

from __future__ import annotations

import functools
import math
import statistics
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations
except ImportError as exc:  # pragma: no cover
    raise ImportError('the MCP server needs the mcp extra: pip install "jomini[mcp]"') from exc

from jomini import Battle, GeneralLaw, LinearLaw, LogarithmicLaw, SquareLaw, catalog, cdb90
from jomini.cdb90 import HistoricalBattle, Side
from jomini.models import LanchesterModel
from jomini.research import combat_power_model, estimate_exponent, relative_effectiveness
from jomini.stochastic import monte_carlo

LawName = Literal["square", "linear", "logarithmic"]
_LAWS: dict[str, tuple[type[LanchesterModel], float]] = {
    "square": (SquareLaw, 2.0),
    "linear": (LinearLaw, 1.0),
    "logarithmic": (LogarithmicLaw, 0.0),
}
_ROLE = {"red": "attacker", "blue": "defender"}
_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def _provenance(
    *keys: str,
) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Append the sources and methods behind a tool's result, with keys for `cite`."""
    refs = catalog.references_for(keys)
    block = {
        "sources": [r.short for r in refs],
        "cite_keys": [r.key for r in refs],
        "jomini_version": catalog.VERSION,
    }

    def decorate(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            out = fn(*args, **kwargs)
            out["provenance"] = block
            return out

        return wrapper

    return decorate


INSTRUCTIONS = """\
Jomini gives you quantitative military history: CDB90, a US Army database of 660 land
battles (1600-1982) with strengths, casualties, durations, commanders, outcomes and
narratives, plus Lanchester attrition models fitted to them.

Conventions:
- In every battle the ATTACKER is "red" and the DEFENDER is "blue".
- Time is in days of active combat. rho is the attacker's per-day effectiveness and beta
  the defender's. "effectiveness_ratio" = rho/beta: how many times as deadly each
  attacking soldier was as each defending soldier.
- Across CDB90, once soldier quality is held fixed, battles follow the linear law (law
  exponent ~0.93, 95% CI 0.81-1.06). The square law, the textbook default, fits history
  poorly. Prefer law="linear" unless you have a reason not to, and say which law you used.
- combat_power explains effectiveness with analysts' ratings (leadership, training,
  surprise, ...). Those ratings were assigned with the outcome known: treat the
  multipliers as associations, not causes.

Research workflow:
- Start with list_datasets: it gives each dataset's coverage, licence and caveats. Datasets:
  CDB90 (660 battles), Livermore (63 Civil War battles), Kursk (daily, both sides), and the
  war in Ukraine (Ukrainian General Staff claims; Oryx visually confirmed losses).
- Time series: get_loss_series, effectiveness_trend (Kalman-smoothed relative
  effectiveness), loss_intensity (smoothed loss rate), fit_law_to_series.
- Air defence: air_defense_raid simulates drone raids against a layered defence fitted
  to Ukraine's nightly reports (or a generic fixed-capacity one: saturation, magazines,
  cost exchange); air_defense_saturation_test checks those reports for saturation.
- Every result carries a "provenance" block; use cite with its cite_keys for BibTeX.
  Report uncertainty (intervals, standard errors) alongside point estimates, and pass on
  the dataset caveats (e.g. Oryx counts only documented losses; General Staff figures are
  one side's claims).

Be honest about limits: figures are historical estimates, Lanchester models capture
attrition only (not manoeuvre, morale collapse or command), and counterfactuals are
model outputs, not facts. Cite the battle's Wikipedia link when you use one.
"""

server = MCPServer(
    "jomini",
    title="Jomini: quantitative military history",
    instructions=INSTRUCTIONS,
)


# --------------------------------------------------------------------------- helpers


def _lookup(battle: int | str) -> HistoricalBattle:
    key: int | str = battle
    if isinstance(battle, str) and battle.strip().isdigit():
        key = int(battle)
    try:
        return cdb90.get(key)
    except (KeyError, TypeError) as exc:
        message = exc.args[0] if exc.args else str(exc)
        raise ToolError(f"{message} (use search_battles to find ids)") from None


def _side_dict(side: Side) -> dict[str, Any]:
    return {
        "force": side.force,
        "actors": list(side.actors),
        "commanders": list(side.commanders),
        "strength": side.strength,
        "casualties": side.casualties,
        "loss_fraction": _round(side.loss_fraction),
        "cavalry": side.cavalry,
        "tanks": side.tanks,
        "artillery": side.artillery,
        "aircraft": side.aircraft,
    }


def _round(x: float | None, digits: int = 4) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return float(f"{x:.{digits}g}")


def _summary(b: HistoricalBattle) -> dict[str, Any]:
    return {
        "id": b.id,
        "name": b.name,
        "date": b.date.isoformat(),
        "war": b.war,
        "attacker": ", ".join(b.attacker.actors) or b.attacker.force,
        "defender": ", ".join(b.defender.actors) or b.defender.force,
        "attacker_strength": b.attacker.strength,
        "defender_strength": b.defender.strength,
        "attacker_casualties": b.attacker.casualties,
        "defender_casualties": b.defender.casualties,
        "winner": b.winner,
    }


def _require_fittable(b: HistoricalBattle) -> None:
    if not b.fittable:
        raise ToolError(
            f"{b.name} (id {b.id}) lacks complete strength/casualty/duration data, "
            "so no model can be fitted to it"
        )


# --------------------------------------------------------------------------- tools


@server.tool(annotations=_READ_ONLY)
@_provenance("cdb90")
def search_battles(
    query: str | None = None,
    war: str | None = None,
    actor: str | None = None,
    commander: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    winner: Literal["attacker", "defender", "draw"] | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    """Find battles in CDB90 (660 land battles, 1600-1982).

    String filters are case-insensitive substrings, except `war`, which matches whole
    words ("World War I" excludes World War II). `query` matches battle names,
    campaigns, locations and Wikipedia titles; `actor` matches a state or army on either
    side (e.g. "Germany", "Confederacy"); `commander` matches either side's commanders.
    Returns summaries; call get_battle for full details.
    """
    found = cdb90.search(
        query,
        war=war,
        actor=actor,
        commander=commander,
        year_from=year_from,
        year_to=year_to,
        winner=winner,
    )
    limit = max(1, min(limit, 200))
    return {"total": len(found), "battles": [_summary(b) for b in found[:limit]]}


@server.tool(annotations=_READ_ONLY)
@_provenance("cdb90")
def get_battle(battle: int | str, include_narrative: bool = True) -> dict[str, Any]:
    """Full record of one battle by CDB90 id or name (e.g. 145 or "Waterloo").

    Includes both sides' forces, commanders, strengths, casualties and equipment, the
    duration in days of active combat, frontage, coordinates, Wikipedia link and, for
    most battles, a narrative account from the original HERO study.
    """
    b = _lookup(battle)
    out: dict[str, Any] = {
        "id": b.id,
        "name": b.name,
        "war": b.war,
        "theater": b.theater,
        "campaign": b.campaign,
        "location": b.location,
        "start": b.start,
        "end": b.end,
        "duration_days": b.duration_days,
        "front_width_km": b.front_width_km,
        "latitude": b.latitude,
        "longitude": b.longitude,
        "winner": b.winner,
        "attacker": _side_dict(b.attacker),
        "defender": _side_dict(b.defender),
        "wikipedia_url": b.wikipedia_url,
        "wikidata_id": b.wikidata_id,
        "fittable": b.fittable,
        "defender_posture": b.defender_posture,
        "attacker_advantages": b.advantages.as_dict(),
    }
    if include_narrative:
        out["narrative"] = b.narrative
        out["significance"] = b.significance
    return out


@server.tool(annotations=_READ_ONLY)
@_provenance("cdb90", "lanchester")
def fit_battle(battle: int | str, law: LawName = "linear") -> dict[str, Any]:
    """Fit a Lanchester law to a battle's historical losses.

    Returns per-day coefficients rho (attacker) and beta (defender) that reproduce the
    recorded casualties over the battle's active duration, and effectiveness_ratio =
    rho/beta: how many times as deadly each attacker was as each defender. Ratios from
    different laws are not comparable; "linear" fits history best overall.
    """
    b = _lookup(battle)
    _require_fittable(b)
    cls, exponent = _LAWS[law]
    model = b.fit(cls)
    ratio = relative_effectiveness(b, exponent)
    stronger = "attacker" if ratio >= 1 else "defender"
    factor = ratio if ratio >= 1 else 1 / ratio
    return {
        "battle": _summary(b),
        "law": law,
        "rho": model.rho,
        "beta": model.beta,
        "time_unit": "days",
        "duration_days": b.duration_days,
        "effectiveness_ratio": _round(ratio),
        "interpretation": (
            f"Under the {law} law each {stronger} soldier was {factor:.2f}x as effective "
            f"as each soldier on the other side."
        ),
    }


@server.tool(annotations=_READ_ONLY)
@_provenance("cdb90", "lanchester", "breakpoints")
def what_if(
    battle: int | str,
    attacker_strength_factor: float = 1.0,
    defender_strength_factor: float = 1.0,
    attacker_effectiveness_factor: float = 1.0,
    defender_effectiveness_factor: float = 1.0,
    duration_days: float | None = None,
    law: LawName = "linear",
) -> dict[str, Any]:
    """Counterfactual: re-fight a historical battle with changed forces.

    Fits `law` to what actually happened, then multiplies each side's starting strength
    and per-soldier effectiveness by the given factors (e.g. attacker_strength_factor=1.2
    for 20% more attackers) and simulates the same duration (or `duration_days`).
    Compares each side's casualties and share of strength lost with history, and gives
    each side's win probability in the historical and changed setups: 1,000 runs with
    breaking points (the losses a side takes before it quits) drawn from a model fitted
    to CDB90 for the battle's era. The model captures attrition and morale breaking,
    not manoeuvre, so present results as estimates.
    """
    b = _lookup(battle)
    _require_fittable(b)
    factors = {
        "attacker_strength_factor": attacker_strength_factor,
        "defender_strength_factor": defender_strength_factor,
        "attacker_effectiveness_factor": attacker_effectiveness_factor,
        "defender_effectiveness_factor": defender_effectiveness_factor,
    }
    for name, value in factors.items():
        if not (math.isfinite(value) and 0 < value <= 100):
            raise ToolError(f"{name} must be in (0, 100]")
    days = b.duration_days if duration_days is None else duration_days
    if not (math.isfinite(days) and days > 0):
        raise ToolError("duration_days must be positive")

    cls, _ = _LAWS[law]
    model = b.fit(cls)
    a0, d0 = b.attacker.strength, b.defender.strength
    assert a0 is not None and d0 is not None  # fittable
    scenario = Battle(a0 * attacker_strength_factor, d0 * defender_strength_factor)
    changed = cls(
        rho=model.rho * attacker_effectiveness_factor,
        beta=model.beta * defender_effectiveness_factor,
    )
    result = changed.simulate(scenario, t_max=days, min_strength=0)
    a_left, d_left = result.remaining

    def side(start: float, left: float) -> dict[str, Any]:
        return {
            "start": round(start),
            "casualties": round(start - left),
            "remaining": round(left),
            "loss_fraction": _round((start - left) / start),
        }

    counter = {
        "attacker": side(scenario.red, a_left),
        "defender": side(scenario.blue, d_left),
        "days_simulated": _round(result.duration),
        "side_destroyed": _ROLE.get(result.loser or ""),
    }
    hist = {
        "attacker": side(a0, a0 - b.attacker.casualties),  # type: ignore[operator]
        "defender": side(d0, d0 - b.defender.casualties),  # type: ignore[operator]
        "days": b.duration_days,
        "winner": b.winner,
    }

    def worse(s: dict[str, Any]) -> str:
        return (
            "attacker"
            if s["attacker"]["loss_fraction"] > s["defender"]["loss_fraction"]
            else "defender"
        )

    def odds(m: LanchesterModel, setup: Battle) -> dict[str, float | None]:
        mc = monte_carlo(m, setup, n=1000, year=b.year, seed=0)
        return {"attacker": _round(mc.p_red, 3), "defender": _round(mc.p_blue, 3)}

    return {
        "battle": _summary(b),
        "law": law,
        "changes": factors | {"duration_days": days},
        "win_probability": {
            "historical_setup": odds(model, b.to_battle()),
            "counterfactual": odds(changed, scenario),
        },
        "historical": hist,
        "counterfactual": counter,
        "side_losing_larger_share": {
            "historical": worse(hist),
            "counterfactual": worse(counter),
        },
    }


@server.tool(annotations=_READ_ONLY)
@_provenance("lanchester", "breakpoints")
def simulate_battle(
    attacker_strength: float,
    defender_strength: float,
    rho: float,
    beta: float,
    law: LawName | None = "linear",
    p: float | None = None,
    q: float | None = None,
    breakpoint: float | None = None,
    max_time: float | None = None,
    win_probability_runs: int = 0,
    year: int = 1900,
) -> dict[str, Any]:
    """Simulate a hypothetical battle from scratch with a Lanchester model.

    rho / beta are the attacker's / defender's effectiveness per time unit (take them
    from fit_battle to stay grounded in history). Give `law`, or set law=None and pass
    Bracken exponents p and q for the general law dR/dt = -beta * B^p * R^q.
    `breakpoint` is the fraction of its strength a side can lose before it breaks
    (e.g. 0.3); without it the fight continues until one side is gone or max_time.
    With win_probability_runs > 0 (up to 5000), also runs that many battles with random
    breaking points fitted to CDB90 for `year` and reports each side's win probability.
    """
    try:
        battle = Battle(attacker_strength, defender_strength)
        model: LanchesterModel
        if law is None:
            if p is None or q is None:
                raise ToolError("with law=None, pass both p and q")
            model = GeneralLaw(rho, beta, p=p, q=q)
        else:
            model = _LAWS[law][0](rho, beta)
        result = model.simulate(battle, t_max=max_time, breakpoint=breakpoint, n_points=11)
        odds = None
        if win_probability_runs:
            if not 0 < win_probability_runs <= 5000:
                raise ToolError("win_probability_runs must be between 1 and 5000")
            mc = monte_carlo(model, battle, n=win_probability_runs, year=year, seed=0)
            odds = {
                "attacker": _round(mc.p_red, 3),
                "defender": _round(mc.p_blue, 3),
                "undecided": _round(mc.p_undecided, 3),
                "runs": mc.n,
                "year": year,
            }
    except (ValueError, TypeError, RuntimeError) as exc:
        raise ToolError(str(exc)) from None
    return {
        "win_probability": odds,
        "winner": _ROLE.get(result.winner or ""),
        "outcome": result.outcome.value,
        "duration": _round(result.duration),
        "attacker": {"start": attacker_strength, "remaining": round(result.remaining[0])},
        "defender": {"start": defender_strength, "remaining": round(result.remaining[1])},
        "trajectory": [
            {"t": _round(t), "attacker": round(r), "defender": round(d)}
            for t, r, d in zip(result.t, result.red, result.blue, strict=True)
        ],
    }


@server.tool(annotations=_READ_ONLY)
def estimate_law(
    war: str | None = None,
    actor: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    query: str | None = None,
    adjust_for_quality: bool = True,
    dataset: Literal["cdb90", "livermore"] = "cdb90",
) -> dict[str, Any]:
    """Estimate which Lanchester law a set of historical battles follows.

    Filters the battles like search_battles (all if no filters) and estimates the
    exponent e = p - q + 1 (2 = square, 1 = linear, 0 = logarithmic) as the value at which
    fighting effectiveness is independent of force ratio (Helmbold's criterion), with a
    95% bootstrap confidence interval. With adjust_for_quality (default; CDB90 only),
    effectiveness only has to be independent of force ratio after accounting for the
    analysts' quality ratings, so a side's numbers aren't credited with its quality.
    dataset="livermore" uses 63 Civil War battles from Livermore (1900) instead, a useful
    replication check: estimates on one war's ~50 battles depend on which battles a
    source chose. Needs at least 10 battles with complete data (more when adjusting).
    """
    from jomini import data

    source = cdb90.battles() if dataset == "cdb90" else data.livermore()
    adjust = adjust_for_quality and dataset == "cdb90"
    pool = (
        cdb90.search(
            query, war=war, actor=actor, year_from=year_from, year_to=year_to, fittable=True
        )
        if dataset == "cdb90"
        else [
            b
            for b in source
            if b.fittable
            and (year_from is None or b.year >= year_from)
            and (year_to is None or b.year <= year_to)
            and (query is None or query.casefold() in b.name.casefold())
        ]
    )
    try:
        est = estimate_exponent(pool, n_boot=500, adjust_for="quality" if adjust else None)
    except ValueError as exc:
        raise ToolError(f"{exc} (matched {len(pool)} battles with complete data)") from None
    refs = catalog.references_for([dataset, "law_estimation"])
    return {
        "dataset": dataset,
        "exponent": _round(est.exponent, 3),
        "ci_95": [_round(est.ci_low, 3), _round(est.ci_high, 3)],
        "n_battles": est.n,
        "nearest_law": est.nearest_law,
        "excluded_laws": [law for law in _LAWS if est.excludes(law)],
        "adjusted_for_quality": adjust,
        "summary": str(est),
        "provenance": {
            "sources": [r.short for r in refs],
            "cite_keys": [r.key for r in refs],
            "jomini_version": catalog.VERSION,
        },
    }


@server.tool(annotations=_READ_ONLY)
@_provenance("cdb90", "law_estimation")
def compare_effectiveness(
    actor: str,
    opponent: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    law: LawName = "linear",
) -> dict[str, Any]:
    """How effective were an actor's soldiers relative to their enemies, battle by battle?

    For every battle with complete data where `actor` fought (optionally only against
    `opponent`), computes the actor's per-soldier effectiveness relative to the other
    side (>1 means the actor's soldiers were deadlier). Returns the geometric mean and
    median ratio, win record, and the most and least effective battles. Useful for
    questions like "how effective was the German army against the Soviets?".
    """
    needle = actor.casefold()
    other = opponent.casefold() if opponent else None
    rows = []
    for b in cdb90.search(actor=actor, year_from=year_from, year_to=year_to, fittable=True):
        attacker_match = any(needle in x.casefold() for x in (*b.attacker.actors, b.attacker.force))
        defender_match = any(needle in x.casefold() for x in (*b.defender.actors, b.defender.force))
        if attacker_match == defender_match:
            continue  # actor on both sides (civil wars) or neither: ambiguous
        role = "attacker" if attacker_match else "defender"
        enemy = b.defender if role == "attacker" else b.attacker
        if other and not any(other in x.casefold() for x in (*enemy.actors, enemy.force)):
            continue
        ratio = relative_effectiveness(b, _LAWS[law][1])
        rows.append((b, role, ratio if role == "attacker" else 1 / ratio))
    if not rows:
        raise ToolError("no battles with complete data match those filters")

    ratios = [r for _, _, r in rows]
    ranked = sorted(rows, key=lambda row: row[2])

    def row(b: HistoricalBattle, role: str, r: float) -> dict[str, Any]:
        return {
            "id": b.id,
            "name": b.name,
            "year": b.year,
            "role": role,
            "effectiveness_ratio": _round(r),
            "won": b.winner == role,
        }

    wins = sum(b.winner == role for b, role, _ in rows)
    return {
        "actor": actor,
        "opponent": opponent,
        "law": law,
        "n_battles": len(rows),
        "geometric_mean_ratio": _round(math.exp(statistics.fmean(math.log(r) for r in ratios))),
        "median_ratio": _round(statistics.median(ratios)),
        "wins": wins,
        "as_attacker": sum(role == "attacker" for _, role, _ in rows),
        "most_effective": [row(*x) for x in ranked[::-1][:5]],
        "least_effective": [row(*x) for x in ranked[:5]],
    }


@server.tool(annotations=_READ_ONLY)
@_provenance("cdb90", "combat_power")
def combat_power(battle: int | str | None = None) -> dict[str, Any]:
    """Which quality factors made soldiers deadlier, across all of CDB90?

    Returns multipliers on the attacker's per-soldier effectiveness for one point of
    advantage in leadership, training, surprise, air superiority, technology and other
    analyst-rated factors, plus the defender's posture (relative to a hasty defence),
    fitted under the quality-adjusted law. With `battle`, also compares the effectiveness
    the factors predict for that battle with what actually happened. The ratings were
    assigned with the outcome known, so these are associations, not causal effects.
    """
    model = combat_power_model()
    out: dict[str, Any] = {
        "exponent": _round(model.exponent, 3),
        "n_battles": model.n,
        "r_squared": _round(model.r_squared, 3),
        "baseline_effectiveness_ratio": _round(model.baseline),
        "multipliers": {
            name: {
                "per_point": _round(m.estimate),
                "ci_95": [_round(m.ci_low), _round(m.ci_high)],
                "significant": m.significant,
            }
            for name, m in sorted(model.multipliers.items(), key=lambda kv: -abs(kv[1].t))
        },
    }
    if battle is not None:
        b = _lookup(battle)
        predicted = model.predict_battle(b)
        actual = relative_effectiveness(b, model.exponent) if b.fittable else None
        out["battle"] = {
            **_summary(b),
            "defender_posture": b.defender_posture,
            "attacker_advantages": b.advantages.as_dict(),
            "predicted_effectiveness_ratio": _round(predicted),
            "actual_effectiveness_ratio": _round(actual),
        }
    return out


@server.tool(annotations=_READ_ONLY)
@_provenance("salvo")
def salvo_exchange(
    attacker_units: float,
    defender_units: float,
    attacker_offense: float,
    attacker_defense: float,
    attacker_staying_power: float,
    defender_offense: float,
    defender_defense: float,
    defender_staying_power: float,
    salvos: int = 10,
    first: Literal["both", "attacker", "defender"] = "both",
) -> dict[str, Any]:
    """Hughes' salvo model for missile, drone-swarm or naval exchanges fired in volleys.

    Per unit: offense = well-aimed shots fired per salvo, defense = incoming shots
    intercepted per salvo, staying_power = hits needed to disable it. Units lost in an
    exchange = max(0, enemy offense - own defense) / staying power. `first` lets one
    side fire an opening salvo unanswered. Returns units left after each salvo and the
    fractional exchange ratio (defender's fractional loss / attacker's, >1 favours the
    attacker) for the opening exchange.
    """
    from jomini.modern import SalvoModel, SalvoSide

    try:
        model = SalvoModel(
            SalvoSide(attacker_offense, attacker_defense, attacker_staying_power),
            SalvoSide(defender_offense, defender_defense, defender_staying_power),
        )
        if not 1 <= salvos <= 1000:
            raise ValueError("salvos must be between 1 and 1000")
        side = {"both": "both", "attacker": "red", "defender": "blue"}[first]
        result = model.simulate(attacker_units, defender_units, salvos=salvos, first=side)  # type: ignore[arg-type]
        fer = model.fractional_exchange_ratio(attacker_units, defender_units)
    except (ValueError, KeyError) as exc:
        raise ToolError(str(exc)) from None
    return {
        "winner": _ROLE.get(result.winner or ""),
        "salvos_fired": result.salvos,
        "attacker_units": [_round(x) for x in result.red],
        "defender_units": [_round(x) for x in result.blue],
        "fractional_exchange_ratio": _round(fer),
    }


# --------------------------------------------------------------------------- research


SeriesSource = Literal["kursk", "ukraine_oryx_events", "ukraine_oryx", "ukraine_general_staff"]


def _series_pair(source: str, category: str, every_days: int) -> tuple[Any, Any, list[str], str]:
    """(red LossSeries-like, blue, side names, note) on a common period grid."""
    from datetime import date as _date

    from jomini import data

    if source == "kursk":
        k = data.kursk()
        if category not in ("casualties", "tanks_lost", "apcs_lost", "artillery_lost"):
            raise ToolError(
                "for kursk, category is casualties, tanks_lost, apcs_lost or artillery_lost"
            )
        dates = k.dates
        red = data.LossSeries(dates, {category: np.cumsum(k.series["german"][category])}, "")
        blue = data.LossSeries(dates, {category: np.cumsum(k.series["soviet"][category])}, "")
        return red, blue, ["german", "soviet"], "Daily; the Germans attack."
    if source in ("ukraine_oryx_events", "ukraine_oryx"):
        loader = data.ukraine_oryx_events if source == "ukraine_oryx_events" else data.ukraine_oryx
        ru, ua = loader("Russia"), loader("Ukraine")
        start: _date = max(ru.dates[0], ua.dates[0])
        end: _date = min(ru.dates[-1], ua.dates[-1])
        for s_ in (ru, ua):
            if category not in s_.cumulative:
                raise ToolError(
                    f"unknown category {category!r}; try one of "
                    + ", ".join(sorted(s_.cumulative)[:25])
                )
        return (
            ru.resample(every_days, start, end),
            ua.resample(every_days, start, end),
            ["russia", "ukraine"],
            f"{every_days}-day periods, {start} to {end}.",
        )
    raise ToolError("two-sided sources are kursk, ukraine_oryx_events and ukraine_oryx")


@server.tool(annotations=_READ_ONLY)
def list_datasets() -> dict[str, Any]:
    """Catalogue of every dataset Jomini can use: coverage, unit of observation, access
    (bundled or fetched from the web at call time), licence, caveats and how to cite it.
    Read the caveats before drawing conclusions from a dataset."""
    return {
        "datasets": [
            {
                "id": d.id,
                "title": d.title,
                "coverage": d.coverage,
                "unit": d.unit,
                "access": d.access,
                "licence": d.licence,
                "python": d.loader,
                "caveats": list(d.caveats),
                "cite": [r.short for r in d.references()],
                "cite_keys": list(d.cite),
            }
            for d in catalog.DATASETS.values()
        ],
        "methods": {
            k: [catalog.REFERENCES[r].short for r in v] for k, v in catalog.METHODS.items()
        },
    }


@server.tool(annotations=_READ_ONLY)
def cite(keys: list[str]) -> dict[str, Any]:
    """BibTeX for datasets (e.g. "cdb90", "kursk"), methods (e.g. "law_estimation",
    "breakpoints", "kalman", "sindy") or reference keys from any tool's provenance.
    Always includes Jomini itself. Verify entries against publishers before formal use."""
    try:
        refs = catalog.references_for([*keys, "jomini"])
    except KeyError as exc:
        raise ToolError(str(exc.args[0])) from None
    return {
        "bibtex": "\n\n".join(r.bibtex() for r in refs),
        "labels": {r.key: r.short for r in refs},
    }


@server.tool(annotations=_READ_ONLY)
def get_loss_series(
    source: SeriesSource,
    category: str,
    side: Literal["russia", "ukraine", "german", "soviet"] = "russia",
    every_days: int = 7,
    max_points: int = 400,
) -> dict[str, Any]:
    """Loss time series per period from a time-series dataset (see list_datasets).

    Categories: kursk -> manpower, casualties, tanks, tanks_lost, apcs, apcs_lost,
    artillery, artillery_lost (side german/soviet, daily). ukraine_general_staff ->
    personnel, tank, APC, "field artillery", drone, ... (Russian side only).
    ukraine_oryx / ukraine_oryx_events -> Oryx classes such as "Tanks" or "All Types".
    Returns period dates, losses per period and cumulative totals.
    """
    from jomini import data

    if not 1 <= every_days <= 365:
        raise ToolError("every_days must be between 1 and 365")
    if source == "kursk":
        if side not in ("german", "soviet"):
            raise ToolError("kursk sides are german and soviet")
        k = data.kursk()
        if category not in k.series[side]:
            raise ToolError(f"kursk categories: {', '.join(k.series[side])}")
        values = k.series[side][category]
        rows = [
            {"date": d.isoformat(), "value": float(v)} for d, v in zip(k.dates, values, strict=True)
        ]
        return {
            "source": source,
            "side": side,
            "category": category,
            "points": rows,
            "note": "Daily values as reported (on-hand counts or losses that day).",
            "provenance": _prov_block(["kursk"]),
        }
    if source == "ukraine_general_staff":
        if side != "russia":
            raise ToolError("ukraine_general_staff only covers Russian losses")
        series = data.ukraine_general_staff()
    elif source in ("ukraine_oryx", "ukraine_oryx_events"):
        if side not in ("russia", "ukraine"):
            raise ToolError("Ukraine war sides are russia and ukraine")
        loader = data.ukraine_oryx if source == "ukraine_oryx" else data.ukraine_oryx_events
        series = loader("Russia" if side == "russia" else "Ukraine")
    else:
        raise ToolError(f"unknown source {source!r}")
    if category not in series.cumulative:
        raise ToolError("unknown category; available: " + ", ".join(sorted(series.cumulative)))
    grid = series.resample(every_days)
    per = grid.daily(category)
    rows = [
        {"period_end": d.isoformat(), "losses": float(x), "cumulative": float(c)}
        for d, x, c in zip(grid.dates, per, grid.cumulative[category], strict=True)
    ]
    return {
        "source": source,
        "side": side,
        "category": category,
        "every_days": every_days,
        "points": rows[-max_points:],
        "truncated": len(rows) > max_points,
        "provenance": _prov_block([source]),
    }


def _prov_block(keys: list[str]) -> dict[str, Any]:
    refs = catalog.references_for(keys)
    return {
        "sources": [r.short for r in refs],
        "cite_keys": [r.key for r in refs],
        "jomini_version": catalog.VERSION,
    }


@server.tool(annotations=_READ_ONLY)
def effectiveness_trend(
    source: Literal["kursk", "ukraine_oryx_events", "ukraine_oryx", "ucdp"],
    category: str = "All Types",
    every_days: int = 28,
    level: float = 0.9,
) -> dict[str, Any]:
    """How one side's per-unit effectiveness relative to the other changed over time.

    Treats log(rho/beta) as a hidden random walk and recovers its path with a Kalman
    smoother from each period's losses (linear law, under which strengths cancel and the
    ratio is the inverse loss-exchange ratio; counting noise plus estimated
    overdispersion). Red is the first side: Germany at Kursk (category casualties,
    tanks_lost, ...), Russia for the Ukraine sources (Oryx classes, e.g. "Tanks").
    source="ucdp" uses UCDP GED battle deaths per side in the Russia-Ukraine conflict
    (personnel, not equipment; category is ignored), from 24 Feb 2022.
    Ratio > 1 means red inflicted more losses per loss suffered. Oryx counts documented
    losses only; documentation rates differ by side and time.
    """
    from jomini.dynamics import effectiveness_over_time

    if not 0.5 <= level < 1:
        raise ToolError("level must be in [0.5, 1)")
    if source == "kursk":
        cat = "casualties" if category == "All Types" else category
        red, blue, sides, note = _series_pair(source, cat, 1)
        rl = np.diff(red.cumulative[cat], prepend=0.0)
        bl = np.diff(blue.cumulative[cat], prepend=0.0)
        dates = red.dates
    elif source == "ucdp":
        from datetime import date as _date

        from jomini import data

        if not 1 <= every_days <= 365:
            raise ToolError("every_days must be between 1 and 365")
        ev = data.ucdp_ged().select(conflict="Russia - Ukraine", start=_date(2022, 2, 24))
        cat = "battle deaths"
        red = ev.daily("deaths_a", start=_date(2022, 2, 24)).resample(every_days)
        blue = ev.daily("deaths_b", start=_date(2022, 2, 24)).resample(every_days)
        rl, bl = red.daily("deaths_a")[1:], blue.daily("deaths_b")[1:]
        dates = red.dates[1:]
        sides = ["russia", "ukraine"]
        note = f"UCDP GED best estimates, {every_days}-day periods from 2022-02-24."
    else:
        if not 1 <= every_days <= 365:
            raise ToolError("every_days must be between 1 and 365")
        cat = category
        red, blue, sides, note = _series_pair(source, cat, every_days)
        rl, bl = red.daily(cat)[1:], blue.daily(cat)[1:]
        dates = red.dates[1:]
    try:
        path = effectiveness_over_time(rl, bl)
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    lo, hi = path.interval(level)
    return {
        "red": sides[0],
        "blue": sides[1],
        "category": cat,
        "note": note,
        "points": [
            {
                "period_end": d.isoformat(),
                "ratio": _round(r),
                "low": _round(a),
                "high": _round(b),
                "red_losses": float(x),
                "blue_losses": float(y),
            }
            for d, r, a, b, x, y in zip(dates, path.ratio(), lo, hi, rl, bl, strict=True)
        ],
        "interval_level": level,
        "step_sd_per_period": _round(path.step_sd),
        "overdispersion_sd": _round(path.overdispersion_sd),
        "provenance": _prov_block(
            [{"ucdp": "ucdp_ged"}.get(source, source), "kalman", "lanchester"]
        ),
    }


@server.tool(annotations=_READ_ONLY)
def loss_intensity(
    source: Literal["ukraine_general_staff", "ukraine_oryx_events", "ukraine_oryx"],
    category: str,
    side: Literal["russia", "ukraine"] = "russia",
    every_days: int = 7,
    level: float = 0.9,
) -> dict[str, Any]:
    """Smoothed underlying loss rate behind a noisy loss series, with a credible band.

    E.g. Russian personnel losses claimed by Ukraine per week, or confirmed Ukrainian tank
    losses per 28 days. Uses a Kalman smoother on log losses (counting noise plus
    estimated overdispersion); negative corrections are treated as missing.
    """
    from jomini.dynamics import intensity_over_time

    series_out = get_loss_series(source, category, side, every_days, max_points=100_000)
    points = series_out["points"][1:]
    losses = np.array([p["losses"] for p in points])
    try:
        path = intensity_over_time(losses)
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    lo, hi = path.interval(level)
    return {
        "source": source,
        "side": side,
        "category": category,
        "every_days": every_days,
        "points": [
            {
                "period_end": p["period_end"],
                "observed": p["losses"],
                "rate": _round(r),
                "low": _round(a),
                "high": _round(b),
            }
            for p, r, a, b in zip(points, path.ratio(), lo, hi, strict=True)
        ],
        "provenance": _prov_block([source, "kalman"]),
    }


@server.tool(annotations=_READ_ONLY)
def fit_law_to_series(
    red: list[float],
    blue: list[float],
    red_losses: list[float],
    blue_losses: list[float],
) -> dict[str, Any]:
    """Fit Bracken's law separately for each side from per-period strengths and losses:
    own_losses = coefficient * enemy^p * own^q (log-linear least squares, with standard
    errors). Use for daily or weekly engagement data where strengths may include
    reinforcements. Large standard errors mean the data can't pin the law down (e.g.
    Kursk: strengths vary too little over 14 days). Set red/blue to the strengths at the
    start of each period.
    """
    from jomini.discovery import fit_power_law_from_losses

    try:
        fit = fit_power_law_from_losses(red, blue, red_losses, blue_losses)
    except ValueError as exc:
        raise ToolError(str(exc)) from None

    def side(s: Any) -> dict[str, Any]:
        return {
            "coefficient": s.coefficient,
            "p_enemy": _round(s.p),
            "p_se": _round(s.p_se),
            "q_own": _round(s.q),
            "q_se": _round(s.q_se),
            "exponent_e": _round(s.exponent),
            "r_squared": _round(s.r_squared),
            "n": s.n,
        }

    return {
        "red": side(fit.red),
        "blue": side(fit.blue),
        "provenance": _prov_block(["lanchester", "kursk_fits"]),
    }


@server.tool(annotations=_READ_ONLY)
def breakpoint_distribution(year: int = 1900) -> dict[str, Any]:
    """How much of its strength a side loses before it breaks, for a given year.

    Log-normal model fitted to CDB90 by maximum likelihood with censoring (the loser's
    losses are its breakpoint; the winner's only bound its own). Returns quantiles of
    the breakpoint for attackers and defenders.
    """
    from jomini.stochastic import default_breakpoints

    bp = default_breakpoints()
    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    return {
        "year": year,
        "attacker": {str(q): _round(bp.quantile(q, year, "attacker"), 3) for q in qs},
        "defender": {str(q): _round(bp.quantile(q, year, "defender"), 3) for q in qs},
        "model": {
            "intercept": bp.intercept,
            "per_century": bp.year_slope,
            "defender_shift": bp.defender_shift,
            "sigma": bp.sigma,
            "n_sides": bp.n,
        },
        "note": "Values are fractions of starting strength lost; 1.0 means fight to the end.",
        "provenance": _prov_block(["cdb90", "breakpoints"]),
    }


@server.tool(annotations=_READ_ONLY)
def conflict_events(
    country: str | None = None,
    conflict: str | None = None,
    start: str | None = None,
    end: str | None = None,
    type_of_violence: Literal[1, 2, 3] | None = None,
    max_months: int = 240,
    dataset: Literal["ged", "candidate"] = "ged",
) -> dict[str, Any]:
    """Monthly events and deaths of organised violence from UCDP GED (1989-2025).

    Filter by country or conflict name (case-insensitive substring, e.g. country="Sudan",
    conflict="Russia - Ukraine"), ISO dates, and type_of_violence (1 state-based,
    2 non-state, 3 one-sided violence against civilians). dataset="candidate" uses UCDP's
    provisional 2026 events (Jan-Sep 2026, incl. the 2026 Iran war: conflict="Iran"); its
    per-side deaths can be badly incomplete, so check both sides before comparing.
    Returns per-month event counts,
    best-estimate deaths, deaths on side A / side B, civilian deaths, plus the main
    conflicts and dyads matched. The first call downloads ~39 MB and takes a few seconds.
    """
    from datetime import date as _date

    from jomini import data

    try:
        s0 = _date.fromisoformat(start) if start else None
        e0 = _date.fromisoformat(end) if end else None
    except ValueError:
        raise ToolError("start and end must be ISO dates like 2022-02-24") from None
    if not (country or conflict):
        raise ToolError("give a country or a conflict name")
    ev = (data.ucdp_ged() if dataset == "ged" else data.ucdp_candidate()).select(
        country=country, conflict=conflict, start=s0, end=e0, type_of_violence=type_of_violence
    )
    if not len(ev):
        raise ToolError("no events match those filters")
    months: dict[str, dict[str, float]] = {}
    for value in ("events", "best", "deaths_a", "deaths_b", "deaths_civilians"):
        series = ev.daily(value, s0, e0)
        per = series.daily(value)
        for d, x in zip(series.dates, per, strict=True):
            key = f"{d.year}-{d.month:02d}"
            months.setdefault(
                key,
                dict.fromkeys(("events", "best", "deaths_a", "deaths_b", "deaths_civilians"), 0.0),
            )[value] += float(x)
    rows = [
        {"month": k, **{kk: round(v, 1) for kk, v in vals.items()}}
        for k, vals in sorted(months.items())
    ][-max_months:]

    def top(field: str) -> list[dict[str, Any]]:
        names, counts = np.unique(ev[field].astype(str), return_counts=True)
        order = np.argsort(-counts)[:5]
        return [{"name": str(names[i]), "events": int(counts[i])} for i in order]

    return {
        "events": len(ev),
        "deaths_best": round(float(np.nansum(ev["best"]))),
        "conflicts": top("conflict_name"),
        "dyads": top("dyad_name"),
        "months": rows,
        "note": "deaths_a / deaths_b are deaths on side A / side B of each event's dyad "
        "(side A is a government in state-based conflicts).",
        "provenance": _prov_block(["ucdp_ged" if dataset == "ged" else "ucdp_candidate"]),
    }


@server.tool(annotations=_READ_ONLY)
def escalation_model(
    country: str | None = None,
    conflict: str | None = None,
    start: str | None = None,
    end: str | None = None,
    measure: Literal["events", "deaths"] = "events",
    knot_days: int = 90,
    dataset: Literal["ged", "candidate"] = "ged",
) -> dict[str, Any]:
    """How strongly violence in a conflict feeds on itself: a self-exciting (Hawkes)
    model fitted to daily UCDP GED counts.

    The branching ratio is the expected number of follow-on events each event triggers
    (0 = no clustering; near 1 = highly self-sustaining); the half-life is how fast that
    excitation fades. Two fits are reported: with a smooth spline background rate (slow
    escalation and lulls are not credited to contagion; in simulations it understates
    the branching ratio by about 10-20%) and with a constant background (slow trends
    inflate it, sometimes to ~1; treat as an upper bound). Use the spline fit as the
    headline. Daily counts are negative binomial (overdispersed).
    """
    from datetime import date as _date

    from jomini import data
    from jomini.escalation import fit_hawkes

    try:
        s0 = _date.fromisoformat(start) if start else None
        e0 = _date.fromisoformat(end) if end else None
    except ValueError:
        raise ToolError("start and end must be ISO dates like 2022-02-24") from None
    if not (country or conflict):
        raise ToolError("give a country or a conflict name")
    ev = (data.ucdp_ged() if dataset == "ged" else data.ucdp_candidate()).select(
        country=country, conflict=conflict, start=s0, end=e0
    )
    if not len(ev):
        raise ToolError("no events match those filters")
    value = "events" if measure == "events" else "best"
    series = ev.daily(value, s0, e0)
    counts = series.daily(value)
    try:
        spline = fit_hawkes(counts, baseline="spline", knot_days=knot_days)
        constant = fit_hawkes(counts, baseline="constant")
    except ValueError as exc:
        raise ToolError(str(exc)) from None

    def report(f: Any) -> dict[str, Any]:
        return {
            "branching_ratio": _round(f.branching_ratio, 3),
            "half_life_days": _round(f.half_life_days, 3),
            "background_per_day": _round(f.baseline_rate),
            "dispersion": _round(f.dispersion),
            "lr_test_p": _round(f.p_value, 3),
            "aic": _round(f.aic, 6),
        }

    return {
        "period": [series.dates[0].isoformat(), series.dates[-1].isoformat()],
        "days": len(series.dates),
        "measure": measure,
        "total": round(float(counts.sum())),
        "spline_background": report(spline),
        "constant_background_upper_bound": report(constant),
        "interpretation": (
            f"Spline fit: each {measure[:-1]} triggers about {spline.branching_ratio:.2f} "
            f"more on average, fading with a half-life of {spline.half_life_days:.1f} days."
        ),
        "provenance": _prov_block(["ucdp_ged" if dataset == "ged" else "ucdp_candidate", "hawkes"]),
    }


class AirDefenseLayer(BaseModel):
    """One ring of a custom air defence."""

    name: str = Field(description="e.g. 'interceptor drones', 'Patriot'")
    outer_km: float = Field(description="Outer edge, km from the target")
    inner_km: float = Field(description="Inner edge, km from the target (0 = at the target)")
    pk: float = Field(description="Chance that one shot kills, 0-1")
    fire_rate: float | None = Field(None, description="Most shots per hour; omit for unlimited")
    engage_rate: float = Field(2.0, description="Shots per hour at a lone drone")
    magazine: float | None = Field(None, description="Shots in stock; omit for unlimited")
    cost_per_shot: float = Field(0.0, description="Dollars per shot")


@server.tool(annotations=_READ_ONLY)
@_provenance("air_defense")
def air_defense_raid(
    raid_sizes: list[float],
    defense: Literal["ukraine", "generic"] = "ukraine",
    layers: list[AirDefenseLayer] | None = None,
    surge: float | None = None,
    magazine_scale: float = 1.0,
    launch_window_hours: float = 1.0,
    attacker_unit_cost: float = 50_000,
    nights: int = 1,
) -> dict[str, Any]:
    """Simulate drone raids against a three-layer air defence (interceptor
    drones/aircraft 150-300 km out, mobile guns 40-150 km, missiles inside 40 km). A
    spatial Lanchester model: the raid flies in and spreads out; each layer's shots
    saturate (Holling type II) and draw down magazines.

    ``defense="ukraine"`` (default) uses the version fitted to Ukraine's nightly Shahed
    reports since Sep 2024: unlimited capacity, surge 0.19, about 15% of a typical
    125-drone raid gets through. ``defense="generic"`` uses round placeholder numbers
    with fixed capacity and finite magazines (shows saturation and depletion). Pass
    ``layers`` to build your own defence instead (any number of rings, any equipment);
    ``defense`` is then ignored.

    Returns leak fraction, kills per layer and cost exchange for each raid size, and
    the elasticity of leakers with respect to raid size (above 1 = saturation). With
    ``surge`` > 0 the defence's capacity grows with raid size as
    (size/100)**surge, modelling forces moved in on warning. With ``nights`` > 1 the
    largest raid size repeats nightly against magazines restocked at 5% a night, to
    show magazine depletion. Layer ranges and costs are placeholders, not any real
    order of battle; the Ukraine fit only pins down overall lethality and surge.
    """
    from jomini.airdefense import AirDefense, Layer, generic_defense, ukraine_defense

    if not raid_sizes or any(s <= 0 or s > 5000 for s in raid_sizes):
        raise ToolError("raid_sizes must be 1 to 5000 attackers each")
    if (surge is not None and not 0 <= surge <= 3) or not 0.05 <= magazine_scale <= 20:
        raise ToolError("surge must be in [0, 3] and magazine_scale in [0.05, 20]")
    if not 1 <= nights <= 60:
        raise ToolError("nights must be 1 to 60")
    if layers:
        if len(layers) > 10:
            raise ToolError("at most 10 layers")
        try:
            model = AirDefense(
                tuple(
                    Layer(
                        lay.name,
                        lay.outer_km,
                        lay.inner_km,
                        pk=lay.pk,
                        fire_rate=math.inf if lay.fire_rate is None else lay.fire_rate,
                        engage_rate=lay.engage_rate,
                        magazine=math.inf if lay.magazine is None else lay.magazine,
                        cost_per_shot=lay.cost_per_shot,
                    )
                    for lay in layers
                )
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from None
    elif defense == "ukraine":
        model = ukraine_defense()
    else:
        model = generic_defense(magazine_scale=magazine_scale)
    if surge is not None:
        model = model.with_surge(surge)
    sizes = sorted(set(raid_sizes))
    raids = []
    for size in sizes:
        r = model.raid(size, launch_window=launch_window_hours,
                         attacker_unit_cost=attacker_unit_cost)  # fmt: skip
        raids.append(
            {
                "launched": size,
                "leaked": _round(r.leaked),
                "leak_fraction": _round(r.leak_fraction, 3),
                "layers": [
                    {
                        "layer": x.name,
                        "kills": _round(x.kills),
                        "shots": _round(x.shots),
                        "magazine_left": _round(x.magazine_left),
                    }
                    for x in r.layers
                ],
                "cost_exchange": _round(r.cost_exchange, 3),
            }
        )
    out: dict[str, Any] = {
        "defense": "custom" if layers else defense,
        "raids": raids,
        "surge": model.layers[0].surge,
    }
    if len(sizes) > 1:
        _, _, el = model.leakage_curve(sizes, launch_window=launch_window_hours)
        out["leak_elasticity"] = [_round(float(e), 3) for e in el]
    if nights > 1:
        stocks = {x.name: x.magazine for x in model.layers}
        restock = {k: 0.05 * v for k, v in stocks.items() if math.isfinite(v)}
        camp = model.campaign([sizes[-1]] * nights, restock=restock,
                                launch_window=launch_window_hours)  # fmt: skip
        out["campaign"] = {
            "raid_size": sizes[-1],
            "leak_fraction_by_night": [_round(float(f), 3) for f in camp.leak_fraction],
            "magazine_empty_after_night": {
                k: (None if (i := camp.empty_after(k)) is None else i + 1) for k in restock
            },
        }
    return out


@server.tool(annotations=_READ_ONLY)
@_provenance("ukraine_air_raids", "air_defense")
def air_defense_saturation_test(
    weapon: Literal["drone", "cruise", "ballistic", "all"] = "drone",
    start: str | None = None,
    end: str | None = None,
    min_launched: int = 5,
) -> dict[str, Any]:
    """Does Ukraine's air defence saturate? Tests whether the share of attackers stopped
    falls (saturation) or rises with raid size, comparing nights within the same month
    (beta-binomial regression with month effects), from the Ukrainian Air Force's
    nightly reports.

    Theory: if each attacker's chance of being shot down can only fall as more attackers
    are present (fixed capacity, finite magazines), interception can't improve with raid
    size, so ``slope`` <= 0 and leak elasticity >= 1. A significantly positive slope
    rejects every such model and points to surge on warning, decoys in bigger raids,
    routing or reporting. Data are the defender's own claims (see caveats).
    """
    from datetime import date as _date

    from jomini import data
    from jomini.airdefense import saturation_test

    try:
        s0 = _date.fromisoformat(start) if start else None
        e0 = _date.fromisoformat(end) if end else None
    except ValueError:
        raise ToolError("start and end must be ISO dates like 2025-10-01") from None
    nights = data.ukraine_air_raids(weapon).select(start=s0, end=e0, min_launched=min_launched)
    if len(nights) < 20:
        raise ToolError("fewer than 20 nights match; widen the dates or lower min_launched")
    try:
        res = saturation_test(nights.launched, nights.stopped, nights.months)
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    lo, hi = res.interval
    share = nights.stopped.sum() / nights.launched.sum()
    return {
        "weapon": weapon,
        "period": [nights.dates[0].isoformat(), nights.dates[-1].isoformat()],
        "nights": res.nights,
        "months": res.periods,
        "launched": round(float(nights.launched.sum())),
        "stopped_share": _round(float(share), 3),
        "slope": _round(res.slope, 3),
        "slope_95ci": [_round(lo, 3), _round(hi, 3)],
        "leak_elasticity": _round(res.leak_elasticity, 3),
        "leak_elasticity_se": _round(res.leak_elasticity_se, 3),
        "verdict": res.verdict,
        "interpretation": (
            f"Within a month, doubling a raid multiplies the odds an attacker is stopped by "
            f"{2**res.slope:.2f} and the number that get through by about "
            f"{2**res.leak_elasticity:.2f} (proportional would be 2)."
        ),
    }


def main() -> None:
    """Entry point for the ``jomini-mcp`` command (stdio transport)."""
    server.run()


if __name__ == "__main__":
    main()
