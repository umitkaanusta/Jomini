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
        "list_datasets",
        "cite",
        "get_loss_series",
        "effectiveness_trend",
        "loss_intensity",
        "fit_law_to_series",
        "breakpoint_distribution",
        "conflict_events",
        "escalation_model",
        "air_defense_raid",
        "air_defense_saturation_test",
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


def test_research_tools_listed_with_provenance() -> None:
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {
        "list_datasets",
        "cite",
        "get_loss_series",
        "effectiveness_trend",
        "loss_intensity",
        "fit_law_to_series",
        "breakpoint_distribution",
    } <= names
    out = call("what_if", battle="Waterloo")
    assert "helmbold1971" in out["provenance"]["cite_keys"]
    assert out["provenance"]["jomini_version"]


def test_catalog_and_cite_tools() -> None:
    cat = call("list_datasets")
    ids = {d["id"] for d in cat["datasets"]}
    assert {"cdb90", "kursk", "livermore", "ukraine_general_staff"} <= ids
    assert all(d["caveats"] for d in cat["datasets"])
    bib = call("cite", keys=["kursk", "kalman"])
    assert "@article{lucas2004" in bib["bibtex"] and "jomini" in bib["labels"]
    with pytest.raises(ToolError):
        call("cite", keys=["narnia"])


def test_kursk_series_tools() -> None:
    s = call("get_loss_series", source="kursk", category="tanks_lost", side="soviet")
    assert s["points"][7] == {"date": "1943-07-12", "value": 414.0}
    trend = call("effectiveness_trend", source="kursk")
    assert len(trend["points"]) == 14 and all(p["ratio"] > 1 for p in trend["points"])
    assert trend["points"][0]["low"] < trend["points"][0]["ratio"] < trend["points"][0]["high"]
    with pytest.raises(ToolError):
        call("get_loss_series", source="kursk", category="tanks_lost", side="russia")
    with pytest.raises(ToolError):
        call("effectiveness_trend", source="kursk", category="morale")


def test_estimate_law_on_livermore_and_breakpoints() -> None:
    out = call("estimate_law", dataset="livermore")
    assert out["n_battles"] == 48 and not out["adjusted_for_quality"]
    assert "livermore1900" in out["provenance"]["cite_keys"]
    bp = call("breakpoint_distribution", year=1944)
    assert bp["attacker"]["0.1"] < bp["attacker"]["0.5"] < bp["attacker"]["0.9"]


def test_fit_law_tool_reports_uncertainty() -> None:
    from jomini import data

    k = data.kursk()
    g, s = k.series["german"], k.series["soviet"]
    out = call(
        "fit_law_to_series",
        red=list(g["manpower"]),
        blue=list(s["manpower"]),
        red_losses=list(g["casualties"]),
        blue_losses=list(s["casualties"]),
    )
    assert out["red"]["q_se"] > 3  # not identified, and the tool says so
    with pytest.raises(ToolError):
        call("fit_law_to_series", red=[1, 2], blue=[1, 2], red_losses=[0, 0], blue_losses=[0, 0])


def test_ukraine_tools_with_stubbed_data(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import date, timedelta

    import numpy as np

    from jomini import data

    days = tuple(date(2022, 3, 1) + timedelta(days=i) for i in range(200))
    rng = np.random.default_rng(0)

    def fake_events(side: str, **_: object) -> data.LossSeries:
        rate = 12.0 if side == "Russia" else 4.0  # Russia loses 3x as much
        daily = rng.poisson(rate, len(days)).astype(float)
        return data.LossSeries(
            days, {"All Types": np.cumsum(daily), "Tanks": np.cumsum(daily / 4)}, "stub"
        )

    def fake_gs(**_: object) -> data.LossSeries:
        return data.LossSeries(days, {"personnel": np.cumsum(np.full(len(days), 500.0))}, "stub")

    monkeypatch.setattr(data, "ukraine_oryx_events", fake_events)
    monkeypatch.setattr(data, "ukraine_general_staff", fake_gs)
    trend = call("effectiveness_trend", source="ukraine_oryx_events", every_days=14)
    assert trend["red"] == "russia" and len(trend["points"]) >= 10
    assert all(0.2 < p["ratio"] < 0.5 for p in trend["points"])  # ~1/3
    with pytest.raises(ToolError, match="unknown category"):
        call("effectiveness_trend", source="ukraine_oryx_events", category="Battleships")
    rate = call(
        "loss_intensity", source="ukraine_general_staff", category="personnel", every_days=7
    )
    assert all(abs(p["rate"] - 3500) / 3500 < 0.05 for p in rate["points"])
    series = call(
        "get_loss_series",
        source="ukraine_oryx_events",
        category="All Types",
        side="ukraine",
        every_days=28,
    )
    assert series["points"][-1]["cumulative"] > 0
    with pytest.raises(ToolError):
        call(
            "get_loss_series", source="ukraine_general_staff", category="personnel", side="ukraine"
        )


def test_conflict_tools_with_stubbed_ged(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import date

    import numpy as np

    from jomini import data
    from jomini.escalation import simulate_hawkes

    n = 400
    counts = simulate_hawkes(n, baseline_rate=3, branching_ratio=0.5, decay=0.7, seed=3)
    days = np.repeat(np.arange(n), counts.astype(int)) + (date(2022, 1, 1) - date(1970, 1, 1)).days
    m = days.size
    ged = data.ConflictEvents(
        days.astype(np.int64),
        days.astype(np.int64),
        {
            "conflict_name": np.array(["Russia - Ukraine"] * m, dtype=object),
            "dyad_name": np.array(["Russia - Ukraine"] * m, dtype=object),
            "side_a": np.array(["Russia"] * m, dtype=object),
            "side_b": np.array(["Ukraine"] * m, dtype=object),
            "country": np.array(["Ukraine"] * m, dtype=object),
        },
        {
            "deaths_a": np.full(m, 3.0),
            "deaths_b": np.full(m, 1.0),
            "deaths_civilians": np.zeros(m),
            "deaths_unknown": np.zeros(m),
            "best": np.full(m, 4.0),
            "low": np.full(m, 4.0),
            "high": np.full(m, 4.0),
            "latitude": np.zeros(m),
            "longitude": np.zeros(m),
            "type_of_violence": np.ones(m),
            "date_prec": np.ones(m),
        },
    )
    monkeypatch.setattr(data, "ucdp_ged", lambda **_: ged)
    ev = call("conflict_events", conflict="Russia - Ukraine")
    assert ev["events"] == m and ev["months"][0]["month"] == "2022-01"
    assert ev["months"][0]["deaths_a"] == pytest.approx(3 * ev["months"][0]["events"], rel=1e-6)
    esc = call("escalation_model", conflict="Russia - Ukraine")
    assert esc["constant_background_upper_bound"]["branching_ratio"] > 0.2
    assert 0 <= esc["spline_background"]["branching_ratio"] < 1
    assert "hawkes1971" in esc["provenance"]["cite_keys"]
    trend = call("effectiveness_trend", source="ucdp", every_days=30)
    assert all(abs(p["ratio"] - 1 / 3) < 0.05 for p in trend["points"] if p["red_losses"] > 50)
    with pytest.raises(ToolError):
        call("conflict_events")
    with pytest.raises(ToolError):
        call("conflict_events", country="Ukraine", start="last tuesday")


def test_air_defense_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    raid = call("air_defense_raid", raid_sizes=[50, 100, 200], defense="generic", nights=5)
    assert [r["launched"] for r in raid["raids"]] == [50, 100, 200]
    assert all(e > 1 for e in raid["leak_elasticity"])
    assert len(raid["campaign"]["leak_fraction_by_night"]) == 5
    assert "randow2026" in raid["provenance"]["cite_keys"]
    surged = call("air_defense_raid", raid_sizes=[50, 200], defense="generic", surge=1.0)
    assert surged["raids"][1]["leak_fraction"] < surged["raids"][0]["leak_fraction"]
    fitted = call("air_defense_raid", raid_sizes=[60, 125, 500])
    assert fitted["defense"] == "ukraine" and fitted["surge"] == pytest.approx(0.19)
    shares = [r["leak_fraction"] for r in fitted["raids"]]
    assert shares[0] > shares[1] > shares[2] and 0.1 < shares[1] < 0.2
    with pytest.raises(ToolError):
        call("air_defense_raid", raid_sizes=[])

    from datetime import date, timedelta

    import numpy as np

    from jomini import data

    rng = np.random.default_rng(0)
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(120)]
    launched = np.round(np.exp(rng.normal(4.5, 0.6, 120)))
    stopped = np.round(launched * 0.85)
    nights = data.RaidNights(tuple(days), launched, stopped, stopped, np.full(120, np.nan), "drone")
    monkeypatch.setattr(data, "ukraine_air_raids", lambda weapon: nights)
    sat = call("air_defense_saturation_test")
    assert sat["nights"] == 120 and sat["months"] == 4
    assert abs(sat["slope"]) < 0.2 and sat["stopped_share"] == pytest.approx(0.85, abs=0.01)
    assert "ivaniuk_missiles" in sat["provenance"]["cite_keys"]
    with pytest.raises(ToolError):
        call("air_defense_saturation_test", start="yesterday")
