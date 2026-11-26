import pytest

from jomini import equipment as eq

UKRAINE_CITY = {
    "mobile gun team": 30,
    "Gepard": 4,
    "interceptor drone team": 10,
    "jamming": 2,
    "F-16": 2,
    "Mi-8 / Mi-24": 2,
    "IRIS-T SLM": 1,
    "NASAMS": 1,
    "Patriot PAC-3": 1,
}


def test_lookup_by_name_and_alias() -> None:
    assert eq.attacker("geran-2").name == "Shahed-136"
    assert eq.attacker("TB2").kind == "combat drone"
    assert eq.defender("patriot").name == "Patriot PAC-3"
    with pytest.raises(KeyError):
        eq.attacker("Death Star")
    with pytest.raises(KeyError):
        eq.defender("force field")


def test_catalogue_is_consistent() -> None:
    for items in (eq.ATTACKERS, eq.DEFENDERS):
        keys = [k.casefold() for x in items for k in (x.name, *x.aliases)]
        assert len(keys) == len(set(keys))
    assert {a.kind for a in eq.ATTACKERS} <= set(eq.ATTACKER_KINDS)
    assert {d.kind for d in eq.DEFENDERS} <= set(eq.DEFENDER_KINDS)
    for kind in eq.ATTACKER_KINDS:
        assert any(eq.kill_chance(d, kind) > 0 for d in eq.DEFENDER_KINDS), kind
    assert all(
        0 <= eq.kill_chance(d, a) <= 0.99 for d in eq.DEFENDER_KINDS for a in eq.ATTACKER_KINDS
    )


def test_calibrated_to_ukraine() -> None:
    night = [
        eq.wave("Shahed-136", 125, launch_window=2.0),
        eq.wave("Kh-101", 10, launch_at=1.0, launch_window=0.5),
    ]
    r = eq.defense(UKRAINE_CITY).strike(night)
    assert 1 - r.leak_fraction("Shahed-136") == pytest.approx(0.87, abs=0.02)
    assert 1 - r.leak_fraction("Kh-101") == pytest.approx(0.78, abs=0.03)


def test_patriot_holds_fire_on_drones_unless_told() -> None:
    drones = [eq.wave("Shahed-136", 50, launch_window=1.0)]
    held = eq.defense({"Patriot PAC-3": 1}).strike(drones)
    assert held.shots.sum() == 0
    free = eq.AirDefense((eq.layer("Patriot PAC-3", targets="all"),)).strike(drones)
    assert free.shots.sum() > 0
    with pytest.raises(ValueError):
        eq.layer("Patriot PAC-3", targets=["unicorns"])


def test_patriot_stops_ballistic_missiles_and_guns_cannot() -> None:
    iskanders = [eq.wave("Iskander-M", 4, launch_window=0.05)]
    assert eq.defense({"Gepard": 10}).strike(iskanders).leak_fraction("Iskander-M") == 1.0
    assert eq.defense({"Patriot PAC-3": 1}).strike(iskanders).leak_fraction("Iskander-M") < 0.4


def test_stealth_gets_through_more() -> None:
    d = eq.defense({"S-400": 1, "Su-35": 4})
    f35 = d.strike([eq.wave("F-35 (strike)", 8, launch_window=0.1)]).leak_fraction("F-35 (strike)")
    f16 = d.strike([eq.wave("F-16 (strike)", 8, launch_window=0.1)]).leak_fraction("F-16 (strike)")
    assert f35 > 2 * f16


def test_layer_options() -> None:
    lay = eq.layer("F-16", 3, deployed=(50, 400))
    assert lay.name == "F-16 x3" and lay.outer == 300 and lay.inner == 50
    assert lay.fire_rate == pytest.approx(3 * eq.layer("F-16").fire_rate)
    with pytest.raises(ValueError):
        eq.layer("F-16", 0)
    with pytest.raises(ValueError):
        eq.defense({})


def test_same_airspace_whatever_the_defence() -> None:
    # Attackers enter 300 km out for catalogue defences, so a short-range defence
    # doesn't make the strike start closer.
    short, long = eq.defense({"Gepard": 1}), eq.defense({"F-16": 1})
    wave = [eq.wave("Shahed-136", 10)]
    assert short.edge == long.edge == 300
    assert eq.AirDefense(short.layers).strike(wave).launched[0] == 10
    with pytest.raises(ValueError):
        eq.AirDefense(short.layers, edge=-1)


def test_cheap_layers_save_the_batteries() -> None:
    strike = [
        eq.wave("Shahed-136", 100, launch_window=1.5),
        eq.wave("Iskander-M", 10, launch_at=2.5, launch_window=0.05),
    ]
    batteries = [eq.layer("NASAMS", targets="all"), eq.layer("Patriot PAC-3", targets="all")]
    cheap = [eq.layer("Gepard", 6), eq.layer("mobile gun team", 30)]
    alone = eq.AirDefense(tuple(batteries), edge=300).strike(strike)
    behind = eq.AirDefense((*batteries, *cheap), edge=300).strike(strike)
    assert alone.leak_fraction("Iskander-M") > 0.9 > 0.6 > behind.leak_fraction("Iskander-M")
