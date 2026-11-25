import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

pytest.importorskip("anthropic")
pytest.importorskip("mcp")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

import jomini_bench as jb

from jomini import cdb90


class FakeClient:
    """Replays scripted responses and records the requests it receives."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses, self.requests = list(responses), []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self.responses.pop(0)


def _msg(stop: str, *blocks: Any) -> Any:
    return NS(stop_reason=stop, content=list(blocks), usage=NS(input_tokens=100, output_tokens=50))


def _answer(p: float = 0.7) -> Any:
    body = {
        "reasoning": "x",
        "p_attacker_wins": p,
        "attacker_casualties": 900,
        "defender_casualties": 1500,
    }
    return _msg("end_turn", NS(type="text", text=json.dumps(body)))


def test_split_is_stable_and_disjoint() -> None:
    train, test = jb.split()
    assert len(test) == 149 and len(train) == 453
    assert not {b.id for b in train} & {b.id for b in test}
    assert jb.split()[1][0].id == test[0].id


def test_anonymised_items_hide_identity() -> None:
    b = cdb90.get("Waterloo")
    anon = jb.describe(b, "anon")
    for leak in ("Waterloo", "France", "Napoleon", "Belgium", "1815", "Hundred Days"):
        assert leak not in anon
    assert "1810s" in anon and "68,265" in anon
    assert "Waterloo" in jb.describe(b, "named")
    with pytest.raises(ValueError):
        jb.describe(b, "other")


def test_runner_plain_condition() -> None:
    client = FakeClient([_answer(0.7)])
    p = jb.run_item(
        client,
        cdb90.get(145),
        model="m",
        variant="anon",
        condition="plain",
        tools=None,
        effort="low",
    )
    assert (p.p_attacker_wins, p.attacker_casualties, p.error) == (0.7, 900, "")
    req = client.requests[0]
    assert "tools" not in req and req["fallbacks"] == "default"
    assert req["output_config"]["format"]["type"] == "json_schema"


def test_runner_executes_safe_tools() -> None:
    tools = jb.jomini_tools()
    assert {t["name"] for t in tools[0]} == set(jb.SAFE_TOOLS)
    cp = next(t for t in tools[0] if t["name"] == "combat_power")
    assert "battle" not in cp["input_schema"]["properties"]  # no outcome leak
    call = NS(type="tool_use", id="t1", name="estimate_law", input={"war": "World War II"})
    client = FakeClient([_msg("tool_use", call), _answer(0.4)])
    p = jb.run_item(
        client,
        cdb90.get(145),
        model="m",
        variant="anon",
        condition="tools",
        tools=tools,
        effort="low",
    )
    assert p.tool_calls == 1 and p.p_attacker_wins == 0.4
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "t1" and "exponent" in result["content"]


def test_runner_reports_refusals_and_bad_answers() -> None:
    refused = jb.run_item(
        FakeClient([_msg("refusal")]),
        cdb90.get(145),
        model="m",
        variant="anon",
        condition="plain",
        tools=None,
        effort="low",
    )
    assert refused.error == "refusal"
    bad = jb.run_item(
        FakeClient([_msg("end_turn", NS(type="text", text="not json"))]),
        cdb90.get(145),
        model="m",
        variant="anon",
        condition="plain",
        tools=None,
        effort="low",
    )
    assert bad.error.startswith("bad answer")


def test_scoring() -> None:
    w = cdb90.get(145)  # defender won
    good = jb.Prediction(145, 0.1, w.attacker.casualties, w.defender.casualties, "x")
    bad = jb.Prediction(145, 0.9, 10, 10, "x")
    assert jb.score([good])["accuracy"] == 1 and jb.score([good])["casualty_log_error"] == 0
    assert jb.score([bad])["brier"] > jb.score([good])["brier"]
    assert jb.score([])["n"] == 0
