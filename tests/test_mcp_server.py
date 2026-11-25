import asyncio
from typing import Any

import pytest

pytest.importorskip("mcp")

from mcp.server.mcpserver.exceptions import ToolError

from jomini.mcp_server import server


def call(name: str, **args: Any) -> dict[str, Any]:
    result = asyncio.run(server.call_tool(name, args))
    assert not result.is_error
    data = result.structured_content
    assert isinstance(data, dict)
    return data


def test_lists_all_tools_as_read_only() -> None:
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {
        "search_battles",
        "get_battle",
        "fit_battle",
        "what_if",
        "simulate_battle",
        "estimate_law",
        "compare_effectiveness",
        "combat_power",
        "salvo_exchange",
    }
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)


def test_search_and_get() -> None:
    found = call("search_battles", commander="napoleon", limit=3)
    assert found["total"] > 3 and len(found["battles"]) == 3
    battle = call("get_battle", battle="Waterloo")
    assert battle["id"] == 145 and battle["winner"] == "defender"
    assert "Napoleon" in battle["narrative"]
    assert call("get_battle", battle="145", include_narrative=False).get("narrative") is None


def test_lookup_errors_are_helpful() -> None:
    with pytest.raises(ToolError, match=r"ambiguous.*search_battles"):
        call("get_battle", battle="Kursk")
    with pytest.raises(ToolError, match="did you mean Waterloo"):
        call("get_battle", battle="waterlo")


def test_fit_battle() -> None:
    out = call("fit_battle", battle=145, law="square")
    assert out["effectiveness_ratio"] == pytest.approx(2.038, rel=1e-3)
    assert "attacker soldier was 2.04x" in out["interpretation"]


def test_what_if_identity_reproduces_history() -> None:
    out = call("what_if", battle="Waterloo")
    hist, counter = out["historical"], out["counterfactual"]
    for side in ("attacker", "defender"):
        assert counter[side]["casualties"] == pytest.approx(hist[side]["casualties"], abs=1)


def test_what_if_more_attackers_inflicts_more_losses() -> None:
    out = call("what_if", battle="Waterloo", attacker_strength_factor=1.3)
    assert (
        out["counterfactual"]["defender"]["casualties"]
        > out["historical"]["defender"]["casualties"]
    )
    with pytest.raises(ToolError, match="factor"):
        call("what_if", battle="Waterloo", attacker_strength_factor=0)


def test_simulate_battle() -> None:
    out = call(
        "simulate_battle",
        attacker_strength=20_000,
        defender_strength=30_000,
        rho=0.015,
        beta=0.012,
        law="square",
    )
    assert out["winner"] == "defender" and len(out["trajectory"]) == 11
    general = call(
        "simulate_battle",
        attacker_strength=1_000,
        defender_strength=1_000,
        rho=0.01,
        beta=0.02,
        law=None,
        p=1,
        q=0,
        breakpoint=0.3,
    )
    assert general["outcome"] == "breakpoint"
    with pytest.raises(ToolError, match="p and q"):
        call("simulate_battle", attacker_strength=1, defender_strength=1, rho=1, beta=1, law=None)
    with pytest.raises(ToolError):
        call("simulate_battle", attacker_strength=-5, defender_strength=10, rho=1, beta=1)


def test_estimate_law_and_effectiveness() -> None:
    law = call("estimate_law", war="world war ii")
    assert "square" in law["excluded_laws"] and law["n_battles"] > 100
    with pytest.raises(ToolError, match="need at least"):
        call("estimate_law", query="waterloo")
    eff = call("compare_effectiveness", actor="Germany", opponent="USSR")
    assert eff["n_battles"] > 10 and eff["geometric_mean_ratio"] > 1
    with pytest.raises(ToolError):
        call("compare_effectiveness", actor="Atlantis")


def test_combat_power_tool() -> None:
    out = call("combat_power", battle="Waterloo")
    assert out["multipliers"]["leadership"]["significant"]
    assert next(iter(out["multipliers"])) == "leadership"  # strongest factor first
    assert out["battle"]["attacker_advantages"]["surprise"] == -2
    assert out["battle"]["predicted_effectiveness_ratio"] > 0
    assert call("get_battle", battle=145)["defender_posture"] == "hasty"


def test_estimate_law_quality_adjustment() -> None:
    adjusted = call("estimate_law")
    plain = call("estimate_law", adjust_for_quality=False)
    assert adjusted["adjusted_for_quality"] and not plain["adjusted_for_quality"]
    assert adjusted["exponent"] > plain["exponent"]


def test_win_probabilities() -> None:
    out = call("what_if", battle="Waterloo", attacker_strength_factor=1.5)
    hist, cf = out["win_probability"]["historical_setup"], out["win_probability"]["counterfactual"]
    assert 0 < hist["attacker"] < cf["attacker"] < 1
    assert hist["attacker"] + hist["defender"] == pytest.approx(1, abs=0.01)
    sim = call(
        "simulate_battle",
        attacker_strength=10_000,
        defender_strength=10_000,
        rho=1e-5,
        beta=1e-5,
        win_probability_runs=500,
        year=1944,
    )
    assert sim["win_probability"]["runs"] == 500
    assert abs(sim["win_probability"]["attacker"] - 0.5) < 0.1
    assert (
        call("simulate_battle", attacker_strength=10, defender_strength=10, rho=0.1, beta=0.1)[
            "win_probability"
        ]
        is None
    )


def test_salvo_tool() -> None:
    args = dict(
        attacker_units=3,
        defender_units=2,
        attacker_offense=4,
        attacker_defense=2,
        attacker_staying_power=1,
        defender_offense=4,
        defender_defense=2,
        defender_staying_power=1,
    )
    out = call("salvo_exchange", **args)
    assert out["winner"] == "attacker" and out["attacker_units"] == [3, 1]
    assert call("salvo_exchange", **args, first="defender")["winner"] == "defender"
    with pytest.raises(ToolError):
        call("salvo_exchange", **{**args, "attacker_staying_power": 0})
