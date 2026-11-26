import io
import zipfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jomini import data
from jomini.airdefense import (
    AirDefense,
    Layer,
    Wave,
    generic_defense,
    missile_defense,
    saturation_test,
    ukraine_defense,
)


def _one_layer(**kw: float) -> AirDefense:
    params = {"pk": 0.5, "fire_rate": 20.0, "engage_rate": 1.0} | kw
    return AirDefense((Layer("sam", 100, 0, **params),), speed=100, diffusion=0)  # type: ignore[arg-type]


def test_small_raids_follow_exponential_survival() -> None:
    # Unsaturated: hazard pk * engage_rate for transit 1 h -> leak fraction exp(-0.5).
    d = _one_layer(fire_rate=1e6)
    r = d.raid(10, launch_window=0)
    assert r.leak_fraction == pytest.approx(np.exp(-0.5), rel=0.03)
    assert r.leaked + r.killed == pytest.approx(10, rel=1e-6)


def test_saturated_layer_kills_at_capacity() -> None:
    # 10 000 attackers; capacity 20 shots/h for 1 h at pk 0.5 -> ~10 kills.
    r = _one_layer().raid(10_000, launch_window=0)
    assert r.killed == pytest.approx(10, rel=0.05)


def test_magazine_and_costs() -> None:
    d = _one_layer(fire_rate=1e3, engage_rate=10, magazine=5, cost_per_shot=100)
    r = d.raid(100, launch_window=0, attacker_unit_cost=10)
    assert r.layers[0].shots == pytest.approx(5) and r.layers[0].magazine_left == pytest.approx(0)
    assert r.defender_cost == pytest.approx(500) and r.cost_exchange == pytest.approx(0.5)


@settings(max_examples=15, deadline=None)
@given(st.floats(1, 300), st.floats(1.1, 4))
def test_fixed_defence_has_no_economies_of_scale(size: float, factor: float) -> None:
    # The module's proposition: without surge, leak fraction never falls with raid size.
    d = generic_defense()
    small, large = d.raid(size), d.raid(size * factor)
    assert large.leak_fraction >= small.leak_fraction - 1e-6


def test_surge_produces_economies_of_scale() -> None:
    _, frac, el = generic_defense().leakage_curve([20, 40, 80, 160])
    assert np.all(np.diff(frac) > 0) and np.all(el > 1)
    _, frac, el = generic_defense().with_surge(1.0).leakage_curve([20, 40, 80, 160])
    assert np.all(np.diff(frac) < 0) and np.all(el < 1)


def test_ukraine_fit_has_economies_of_scale() -> None:
    _, frac, el = ukraine_defense().leakage_curve([30, 60, 125, 250, 500])
    assert np.all(np.diff(frac) < 0) and np.all(el < 1)
    assert 0.12 < ukraine_defense().raid(125).leak_fraction < 0.19
    # Unlimited capacity: engagement stays linear in the number of drones present.
    no_surge = ukraine_defense().with_surge(0.0)
    assert no_surge.raid(50).leak_fraction == pytest.approx(
        no_surge.raid(400).leak_fraction, rel=0.02
    )


def test_campaign_depletes_magazines() -> None:
    d = generic_defense(magazine_scale=0.5)
    camp = d.campaign([200] * 6, restock={"missiles": 2})
    assert camp.empty_after("missiles") is not None
    assert camp.leak_fraction[-1] > camp.leak_fraction[0]
    assert camp.magazines.shape == (6, 3)
    with pytest.raises(ValueError):
        d.campaign([10], restock={"lasers": 1})


def test_validation() -> None:
    with pytest.raises(ValueError):
        Layer("x", 10, 20, pk=0.5, fire_rate=1, engage_rate=1)
    with pytest.raises(ValueError):
        Layer("x", 20, 10, pk=1.5, fire_rate=1, engage_rate=1)
    with pytest.raises(ValueError):
        AirDefense(())
    with pytest.raises(ValueError):
        generic_defense().raid(0)
    with pytest.raises(ValueError):
        generic_defense().raid(10, magazines={"lasers": 3})


def _simulated_nights(slope: float, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    months = np.repeat(np.arange(24), 30)
    base = rng.normal(1.5, 0.4, 24)[months]
    n = np.round(np.exp(rng.normal(4, 0.8, months.size))) + 5
    p = 1 / (1 + np.exp(-(base + slope * (np.log(n) - 4))))
    p = rng.beta(p * 30, (1 - p) * 30)  # overdispersion
    k = rng.binomial(n.astype(int), p)
    return n, k.astype(float), [f"m{m}" for m in months]


@pytest.mark.parametrize("slope", [-0.4, 0.0, 0.5])
def test_saturation_test_recovers_slope(slope: float) -> None:
    n, k, periods = _simulated_nights(slope, seed=int(10 * slope) + 7)
    res = saturation_test(n, k, periods)
    lo, hi = res.interval
    assert lo < slope < hi and res.se < 0.15
    assert res.periods == 24 and res.nights == n.size
    expected = "inconsistent" if slope > 0 else ("consistent" if slope < 0 else "no detectable")
    assert expected in res.verdict


def test_saturation_test_validation() -> None:
    with pytest.raises(ValueError):
        saturation_test([10, 20], [11, 5], ["a", "a"])
    with pytest.raises(ValueError):
        saturation_test([10, 20, 30], [1, 2, 3], ["a", "b", "c"])


_CSV = """time_start,time_end,model,target,launched,destroyed,not_reach_goal,num_hit_location
2025-10-01 18:00,2025-10-02 08:00,Shahed-136/131,Ukraine,100,80,5,7
2025-10-01 18:00,2025-10-02 08:00,Shahed-136/131,south,30,20,,
2025-10-01 18:00,2025-10-02 08:00,Iskander-M,Ukraine,4,1,,
2025-10-02 18:00,2025-10-03 08:00,Geran-4 and Geran-5,west,10,9,,
2025-10-02 18:00,2025-10-03 08:00,Orlan-10,east,3,3,,
2025-10-03,2025-10-03,X-101/X-555 and Kalibr,Ukraine,20,18,0,
2025-10-03,2025-10-03,X-101/X-555 and Iskander-M,Ukraine,5,4,0,
"""


def test_ukraine_air_raids_parsing() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("missile_attacks_daily.csv", _CSV)
    zipped = buf.getvalue()
    drones = data.ukraine_air_raids("drone", fetch=lambda url: zipped)
    assert [d.isoformat() for d in drones.dates] == ["2025-10-01", "2025-10-02"]
    # The nationwide report wins over the overlapping regional one.
    assert list(drones.launched) == [100, 10] and list(drones.stopped) == [85, 9]
    assert drones.hit_locations[0] == 7 and np.isnan(drones.hit_locations[1])
    assert drones.months == ["2025-10", "2025-10"]
    cruise = data.ukraine_air_raids("cruise", fetch=lambda url: _CSV.encode())
    assert list(cruise.launched) == [20]
    assert list(data.ukraine_air_raids("all", fetch=lambda url: zipped).launched) == [104, 10, 25]
    assert len(drones.select(min_launched=50)) == 1
    assert data.weapon_class("Orlan-10 and ZALA") == "other"
    assert data.weapon_class("C-400 and Iskander-M/KN-23") == "ballistic"
    with pytest.raises(ValueError):
        data.ukraine_air_raids("lasers", fetch=lambda url: zipped)  # type: ignore[arg-type]


def test_single_wave_strike_matches_raid() -> None:
    d = generic_defense()
    wave = Wave("drones", 120, d.speed, launch_window=1.0, diffusion=d.diffusion)
    assert d.strike([wave]).leaked[0] == pytest.approx(d.raid(120).leaked, rel=1e-6)


def test_drones_shield_missiles() -> None:
    d = missile_defense()
    missiles = Wave("missiles", 100, 7000, launch_at=2.0, launch_window=0.1)
    drones = Wave("drones", 150, 185, launch_window=1.0, diffusion=50)
    alone, mixed = d.strike([missiles]), d.strike([missiles, drones])
    assert alone.leak_fraction("missiles") < 0.1 < 0.25 < mixed.leak_fraction("missiles")
    # Fighters never shoot at missiles; the upper tier never shoots at drones.
    assert mixed.shots[0, 0] == 0 and mixed.shots[1, 1] == 0
    assert mixed.magazine_left[2] == pytest.approx(0, abs=1e-9)
    assert "missiles" in str(mixed)


def test_strike_validation() -> None:
    with pytest.raises(ValueError):
        Wave("x", 0, 100)
    with pytest.raises(ValueError):
        missile_defense().strike([])
    with pytest.raises(ValueError):
        missile_defense().strike([Wave("a", 1, 100), Wave("a", 2, 100)])
    with pytest.raises(ValueError):
        Layer("x", 10, 0, pk=0.5, fire_rate=1, engage_rate=1, pk_against={"a": 2.0})
