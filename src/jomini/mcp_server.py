"""MCP server that turns Jomini into tools for AI agents ("AI military historian").

Run with ``jomini-mcp`` (needs ``pip install "jomini[mcp]"``). Tools return JSON.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Literal

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations
except ImportError as exc:  # pragma: no cover
    raise ImportError('the MCP server needs the mcp extra: pip install "jomini[mcp]"') from exc

from jomini import Battle, GeneralLaw, LinearLaw, LogarithmicLaw, SquareLaw, cdb90
from jomini.cdb90 import HistoricalBattle, Side
from jomini.models import LanchesterModel
from jomini.research import estimate_exponent, relative_effectiveness

LawName = Literal["square", "linear", "logarithmic"]
_LAWS: dict[str, tuple[type[LanchesterModel], float]] = {
    "square": (SquareLaw, 2.0),
    "linear": (LinearLaw, 1.0),
    "logarithmic": (LogarithmicLaw, 0.0),
}
_ROLE = {"red": "attacker", "blue": "defender"}
_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

INSTRUCTIONS = """\
Jomini gives you quantitative military history: CDB90, a US Army database of 660 land
battles (1600-1982) with strengths, casualties, durations, commanders, outcomes and
narratives, plus Lanchester attrition models fitted to them.

Conventions:
- In every battle the ATTACKER is "red" and the DEFENDER is "blue".
- Time is in days of active combat. rho is the attacker's per-day effectiveness and beta
  the defender's. "effectiveness_ratio" = rho/beta: how many times as deadly each
  attacking soldier was as each defending soldier.
- Across CDB90, battles follow something between the linear and logarithmic laws (law
  exponent ~0.74). The square law, the textbook default, fits history poorly. Prefer
  law="linear" unless you have a reason not to, and say which law you used.

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

    String filters are case-insensitive substrings. `query` matches battle names,
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
    }
    if include_narrative:
        out["narrative"] = b.narrative
        out["significance"] = b.significance
    return out


@server.tool(annotations=_READ_ONLY)
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
    Compares each side's casualties and share of strength lost with history. The side
    losing the larger share usually came off worse, but the model does not predict
    morale collapse or manoeuvre, so present results as estimates.
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

    return {
        "battle": _summary(b),
        "law": law,
        "changes": factors | {"duration_days": days},
        "historical": hist,
        "counterfactual": counter,
        "side_losing_larger_share": {
            "historical": worse(hist),
            "counterfactual": worse(counter),
        },
    }


@server.tool(annotations=_READ_ONLY)
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
) -> dict[str, Any]:
    """Simulate a hypothetical battle from scratch with a Lanchester model.

    rho / beta are the attacker's / defender's effectiveness per time unit (take them
    from fit_battle to stay grounded in history). Give `law`, or set law=None and pass
    Bracken exponents p and q for the general law dR/dt = -beta * B^p * R^q.
    `breakpoint` is the fraction of its strength a side can lose before it breaks
    (e.g. 0.3); without it the fight continues until one side is gone or max_time.
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
    except (ValueError, TypeError, RuntimeError) as exc:
        raise ToolError(str(exc)) from None
    return {
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
) -> dict[str, Any]:
    """Estimate which Lanchester law a set of historical battles follows.

    Filters CDB90 like search_battles (all battles if no filters) and estimates the
    exponent e = p - q + 1 (2 = square, 1 = linear, 0 = logarithmic) as the value at which
    fighting effectiveness is independent of force ratio (Helmbold's criterion), with a
    95% bootstrap confidence interval. Needs at least 10 battles with complete data.
    """
    pool = cdb90.search(
        query, war=war, actor=actor, year_from=year_from, year_to=year_to, fittable=True
    )
    try:
        est = estimate_exponent(pool, n_boot=500)
    except ValueError as exc:
        raise ToolError(f"{exc} (matched {len(pool)} battles with complete data)") from None
    return {
        "exponent": _round(est.exponent, 3),
        "ci_95": [_round(est.ci_low, 3), _round(est.ci_high, 3)],
        "n_battles": est.n,
        "nearest_law": est.nearest_law,
        "excluded_laws": [law for law in _LAWS if est.excludes(law)],
        "summary": str(est),
    }


@server.tool(annotations=_READ_ONLY)
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


def main() -> None:
    """Entry point for the ``jomini-mcp`` command (stdio transport)."""
    server.run()


if __name__ == "__main__":
    main()
