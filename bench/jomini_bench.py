"""Jomini-Bench: can an LLM forecast historical battles, and do Jomini's tools help?

Each item is a decisive CDB90 battle described only by what was known *before* it
was fought (strengths, equipment, defender posture, frontage, era). The model must
give the attacker's win probability and each side's casualties.

Two variants separate reasoning from recall:
  named  battle name, date, war and combatants included
  anon   names, places and nationalities removed, year rounded to the decade

Two conditions measure the value of the tools:
  plain  the model alone
  tools  the model can call Jomini's simulation and research tools (the lookup
         tools are withheld, since they would reveal the outcome)

Usage:
  uv run --group bench python bench/jomini_bench.py items           # show split sizes
  uv run --group bench python bench/jomini_bench.py baselines       # free baselines
  uv run --group bench python bench/jomini_bench.py run --variant anon --condition tools --n 20
  uv run --group bench python bench/jomini_bench.py score bench/results/*.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jomini import LinearLaw, cdb90
from jomini.cdb90 import HistoricalBattle
from jomini.stochastic import BreakpointModel, monte_carlo

RESULTS = Path(__file__).resolve().parent / "results"
TEST_FRACTION = 0.25
DEFAULT_MODEL = "claude-opus-5-5"


# --------------------------------------------------------------------------- items


def is_test(battle: HistoricalBattle) -> bool:
    """Stable train/test split by hashed CDB90 id."""
    h = int(hashlib.sha256(f"jomini-bench:{battle.id}".encode()).hexdigest(), 16)
    return (h % 1000) / 1000 < TEST_FRACTION


def eligible() -> list[HistoricalBattle]:
    return [b for b in cdb90.search(fittable=True) if b.winner in ("attacker", "defender")]


def split() -> tuple[list[HistoricalBattle], list[HistoricalBattle]]:
    pool = eligible()
    return [b for b in pool if not is_test(b)], [b for b in pool if is_test(b)]


def _equipment(side: cdb90.Side) -> str:
    parts = [
        f"{v:,} {name}"
        for name, v in (
            ("cavalry", side.cavalry),
            ("tanks", side.tanks),
            ("artillery pieces", side.artillery),
            ("aircraft", side.aircraft),
        )
        if v
    ]
    return ", ".join(parts) if parts else "none recorded"


def describe(b: HistoricalBattle, variant: str) -> str:
    a, d = b.attacker, b.defender
    posture = {
        "hasty": "a hasty defence",
        "prepared": "a prepared defence",
        "fortified": "a fortified defence",
        "delay": "a delaying action",
        "withdrawal": "a withdrawal",
    }.get(b.defender_posture or "", "an unrecorded posture")
    front = f"{b.front_width_km:g} km" if b.front_width_km else "unrecorded"
    if variant == "named":
        header = (
            f"Battle: {b.name} ({b.date.isoformat()}), {b.war}"
            + (f", {b.theater}" if b.theater else "")
            + (f", {b.location}" if b.location else "")
        )
        att = f"{', '.join(a.actors) or 'unknown'}: {a.force}"
        dfn = f"{', '.join(d.actors) or 'unknown'}: {d.force}"
    elif variant == "anon":
        header = f"An unnamed land battle fought in the {b.year // 10 * 10}s."
        att, dfn = "Side A", "Side B"
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return (
        f"{header}\n"
        f"Attacker ({att}): {a.strength:,} troops; equipment: {_equipment(a)}.\n"
        f"Defender ({dfn}): {d.strength:,} troops; equipment: {_equipment(d)}.\n"
        f"The defender is in {posture}. Frontage: {front}."
    )


@dataclass(frozen=True)
class Prediction:
    battle_id: int
    p_attacker_wins: float
    attacker_casualties: float
    defender_casualties: float
    source: str
    variant: str = ""
    condition: str = ""
    model: str = ""
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


# --------------------------------------------------------------------------- scoring


def score(preds: Iterable[Prediction]) -> dict[str, float]:
    rows = [p for p in preds if not p.error]
    if not rows:
        return {"n": 0}
    by_id = {b.id: b for b in cdb90.battles()}
    p = np.clip([r.p_attacker_wins for r in rows], 1e-3, 1 - 1e-3)
    y = np.array([by_id[r.battle_id].winner == "attacker" for r in rows], float)

    def cas_err(pred: float, actual: int | None) -> float:
        return abs(math.log(max(pred, 1.0) / max(actual or 1, 1)))

    att = [cas_err(r.attacker_casualties, by_id[r.battle_id].attacker.casualties) for r in rows]
    dfn = [cas_err(r.defender_casualties, by_id[r.battle_id].defender.casualties) for r in rows]
    return {
        "n": len(rows),
        "accuracy": float(np.mean((p > 0.5) == y)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        # Median |log(predicted / actual)|: 0.69 means typically off by a factor of 2.
        "casualty_log_error": float(np.median(att + dfn)),
        "errors": len(list(preds)) - len(rows) if isinstance(preds, list) else 0,
    }


# --------------------------------------------------------------------------- baselines


def baseline_base_rate(
    train: list[HistoricalBattle], test: list[HistoricalBattle]
) -> list[Prediction]:
    rate = np.mean([b.winner == "attacker" for b in train])
    fa = np.median([b.attacker.loss_fraction for b in train])
    fd = np.median([b.defender.loss_fraction for b in train])
    return [
        Prediction(
            b.id, float(rate), fa * b.attacker.strength, fd * b.defender.strength, "base_rate"
        )  # type: ignore[operator]
        for b in test
    ]


def _features(b: HistoricalBattle) -> list[float]:
    posture = b.defender_posture or "hasty"
    return [
        math.log(b.attacker.strength / b.defender.strength),  # type: ignore[operator]
        (b.year - 1900) / 100,
        float(posture == "prepared"),
        float(posture == "fortified"),
        float(posture in ("delay", "withdrawal")),
    ]


def baseline_logistic(
    train: list[HistoricalBattle], test: list[HistoricalBattle]
) -> list[Prediction]:
    from scipy.optimize import minimize

    x = np.array([[1.0, *_features(b)] for b in train])
    y = np.array([b.winner == "attacker" for b in train], float)

    def nll(w: np.ndarray) -> float:
        z = x @ w
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(w[1:] ** 2))

    w = minimize(nll, np.zeros(x.shape[1]), method="BFGS").x

    # Casualty fractions: era medians from training battles.
    def era_frac(b: HistoricalBattle, role: str) -> float:
        near = [t for t in train if abs(t.year - b.year) <= 40] or train
        return float(np.median([t.side(role).loss_fraction for t in near]))  # type: ignore[arg-type]

    out = []
    for b in test:
        p = 1 / (1 + math.exp(-float(np.array([1.0, *_features(b)]) @ w)))
        out.append(
            Prediction(
                b.id,
                p,
                era_frac(b, "attacker") * b.attacker.strength,  # type: ignore[operator]
                era_frac(b, "defender") * b.defender.strength,  # type: ignore[operator]
                "logistic",
            )
        )
    return out


def baseline_jomini(
    train: list[HistoricalBattle], test: list[HistoricalBattle]
) -> list[Prediction]:
    """Jomini's own pre-battle forecast: linear law + expected quality + breakpoint Monte Carlo.

    Quality can't be observed before a battle, and in history it is correlated with
    numbers (bigger armies tend to be worse). So the attacker's relative effectiveness
    is set to its *expected* value given force ratio, posture and era: a regression of
    log(rho/beta) under the linear law, fitted on training battles. Its residual spread
    becomes the effectiveness uncertainty in the Monte Carlo, and breaking points come
    from a breakpoint model fitted on training battles too.
    """
    from jomini.research import _arrays, log_effectiveness_ratio

    x = np.array([[1.0, *_features(b)] for b in train])
    y = log_effectiveness_ratio(_arrays(train), 1.0)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    sd = float(np.std(y - x @ coef, ddof=x.shape[1]))
    breakpoints = BreakpointModel.fit(train)
    out = []
    for b in test:
        ratio = math.exp(float(np.array([1.0, *_features(b)]) @ coef))
        model = LinearLaw(rho=1e-6 * ratio, beta=1e-6)  # time scale is irrelevant here
        mc = monte_carlo(
            model,
            b.to_battle(),
            n=1000,
            year=b.year,
            breakpoints=breakpoints,
            effectiveness_sd=sd,
            seed=b.id,
        )
        out.append(
            Prediction(
                b.id,
                mc.p_red,
                float(np.median(mc.red_casualties)),
                float(np.median(mc.blue_casualties)),
                "jomini",
            )
        )
    return out


BASELINES: dict[
    str, Callable[[list[HistoricalBattle], list[HistoricalBattle]], list[Prediction]]
] = {
    "base_rate": baseline_base_rate,
    "logistic": baseline_logistic,
    "jomini": baseline_jomini,
}


# --------------------------------------------------------------------------- LLM runner

SYSTEM = """You are a military historian and operations analyst forecasting the outcome of a \
land battle from the information available before it was fought. Reason quantitatively. \
Give your best calibrated estimate even when information is thin; do not refuse. \
Casualties are killed, wounded, captured and missing over the whole battle."""

TOOLS_NOTE = """You can call Jomini, a library of Lanchester attrition models fitted to 660 \
historical battles (CDB90). Useful tools: estimate_law (which attrition law history follows), \
combat_power (how much posture and other factors change relative effectiveness), and \
simulate_battle (simulate with given strengths and effectiveness; set \
win_probability_runs and year to get win probabilities from breaking points fitted to \
history). The attacker is red, the defender blue. Use the tools to ground your estimate."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "Brief justification."},
        "p_attacker_wins": {"type": "number", "description": "Probability 0-1."},
        "attacker_casualties": {"type": "number"},
        "defender_casualties": {"type": "number"},
    },
    "required": ["reasoning", "p_attacker_wins", "attacker_casualties", "defender_casualties"],
    "additionalProperties": False,
}

# Tools that cannot reveal a battle's recorded outcome.
SAFE_TOOLS = ("simulate_battle", "estimate_law", "combat_power")


def jomini_tools() -> tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]:
    """Anthropic tool definitions and handlers for the leak-free subset of the MCP server."""
    from jomini import mcp_server

    listed = asyncio.run(mcp_server.server.list_tools())
    defs, handlers = [], {}
    for t in listed:
        if t.name not in SAFE_TOOLS:
            continue
        schema = json.loads(json.dumps(t.input_schema))
        if t.name == "combat_power":
            # With a battle argument it would return that battle's actual effectiveness.
            schema["properties"].pop("battle", None)
            schema.pop("required", None)
        defs.append({"name": t.name, "description": t.description or "", "input_schema": schema})
        handlers[t.name] = getattr(mcp_server, t.name)
    return defs, handlers


def run_item(
    client: Any,
    b: HistoricalBattle,
    *,
    model: str,
    variant: str,
    condition: str,
    tools: tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]] | None,
    effort: str,
    max_turns: int = 12,
) -> Prediction:
    import anthropic

    system = SYSTEM + ("\n\n" + TOOLS_NOTE if condition == "tools" else "")
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": describe(b, variant)
            + "\n\nEstimate the probability that the attacker wins, and each side's casualties.",
        }
    ]
    usage_in = usage_out = calls = 0
    kwargs: dict[str, Any] = {}
    if tools:
        kwargs["tools"] = tools[0]
    try:
        for _ in range(max_turns):
            response = client.beta.messages.create(
                model=model,
                max_tokens=16000,
                system=system,
                messages=messages,
                output_config={
                    "effort": effort,
                    "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
                },
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **kwargs,
            )
            usage_in += response.usage.input_tokens
            usage_out += response.usage.output_tokens
            if response.stop_reason == "refusal":
                return Prediction(
                    b.id,
                    math.nan,
                    math.nan,
                    math.nan,
                    "llm",
                    variant,
                    condition,
                    model,
                    calls,
                    usage_in,
                    usage_out,
                    "refusal",
                )
            if response.stop_reason == "tool_use" and tools:
                messages.append({"role": "assistant", "content": response.content})
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    calls += 1
                    try:
                        out = tools[1][block.name](**block.input)
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(out),
                            }
                        )
                    except Exception as exc:  # tool errors go back to the model
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(exc),
                                "is_error": True,
                            }
                        )
                messages.append({"role": "user", "content": results})
                continue
            text = next((blk.text for blk in response.content if blk.type == "text"), "")
            data = json.loads(text)
            return Prediction(
                b.id,
                min(max(float(data["p_attacker_wins"]), 0.0), 1.0),
                max(float(data["attacker_casualties"]), 0.0),
                max(float(data["defender_casualties"]), 0.0),
                "llm",
                variant,
                condition,
                model,
                calls,
                usage_in,
                usage_out,
            )
        error = "max_turns"
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        error = f"bad answer: {exc}"
    except anthropic.APIStatusError as exc:
        error = f"api {exc.status_code}: {exc.message}"
    except anthropic.APIConnectionError as exc:
        error = f"connection: {exc}"
    return Prediction(
        b.id,
        math.nan,
        math.nan,
        math.nan,
        "llm",
        variant,
        condition,
        model,
        calls,
        usage_in,
        usage_out,
        error,
    )


def run(args: argparse.Namespace) -> None:
    import anthropic

    _, test = split()
    items = test[: args.n] if args.n else test
    client = anthropic.Anthropic()
    tools = jomini_tools() if args.condition == "tools" else None
    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = RESULTS / f"{args.model}_{args.variant}_{args.condition}_{stamp}.jsonl"
    preds = []
    with path.open("w") as fh:
        for i, b in enumerate(items, 1):
            p = run_item(
                client,
                b,
                model=args.model,
                variant=args.variant,
                condition=args.condition,
                tools=tools,
                effort=args.effort,
            )
            preds.append(p)
            fh.write(json.dumps(asdict(p)) + "\n")
            fh.flush()
            print(
                f"[{i}/{len(items)}] {b.name}: p={p.p_attacker_wins:.2f} "
                f"tools={p.tool_calls} {p.error}",
                file=sys.stderr,
            )
    print(json.dumps({"file": str(path), **score(preds)}, indent=2))


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("items", help="show split sizes and an example item")
    sub.add_parser("baselines", help="score the API-free baselines on the test split")
    r = sub.add_parser("run", help="run an LLM on the test split (calls the Claude API)")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--variant", choices=("named", "anon"), default="anon")
    r.add_argument("--condition", choices=("plain", "tools"), default="plain")
    r.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    r.add_argument("--n", type=int, default=0, help="only the first N test battles")
    s = sub.add_parser("score", help="score saved result files")
    s.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "items":
        train, test = split()
        print(f"train {len(train)}, test {len(test)}\n")
        print(describe(test[0], "named"), "\n")
        print(describe(test[0], "anon"))
    elif args.cmd == "baselines":
        train, test = split()
        for name, fn in BASELINES.items():
            print(
                f"{name:10s}",
                json.dumps({k: round(v, 3) for k, v in score(fn(train, test)).items()}),
            )
    elif args.cmd == "run":
        run(args)
    elif args.cmd == "score":
        for f in args.files:
            preds = [Prediction(**json.loads(line)) for line in f.read_text().splitlines()]
            print(f.name, json.dumps({k: round(v, 3) for k, v in score(preds).items()}))


if __name__ == "__main__":
    main()
