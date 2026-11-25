import math

import numpy as np
import pytest

from jomini import Battle, GeneralLaw, LinearLaw, SquareLaw
from jomini.discovery import fit_power_law, sindy
from jomini.modern import MixedForceLaw, SalvoModel, SalvoSide

# ------------------------------------------------------------------ salvo


def ships(offense: float = 4, defense: float = 2, staying: float = 1) -> SalvoSide:
    return SalvoSide(offense=offense, defense=defense, staying_power=staying)


def test_salvo_exchange_matches_hughes() -> None:
    m = SalvoModel(ships(), ships())
    # Blue loses (4*3 - 2*2)/1 = 8 -> all 2; red loses (4*2 - 2*3)/1 = 2.
    assert m.exchange(3, 2) == (2, 2)
    r = m.simulate(3, 2)
    assert r.winner == "red" and r.remaining == (1, 0) and r.salvos == 1
    assert m.fractional_exchange_ratio(3, 2) == pytest.approx((2 / 2) / (2 / 3))


def test_salvo_defence_absorbs_and_first_strike_decides() -> None:
    m = SalvoModel(ships(offense=2, defense=3), ships(offense=2, defense=3))
    assert m.exchange(5, 5) == (0, 0)  # defences soak everything
    assert m.simulate(5, 5, salvos=3).winner is None
    strong = SalvoModel(ships(), ships())
    assert strong.simulate(3, 2, first="blue").winner == "blue"  # surprise reverses outcome


def test_salvo_validation() -> None:
    with pytest.raises(ValueError):
        SalvoSide(offense=-1, defense=0, staying_power=1)
    with pytest.raises(ValueError):
        SalvoSide(offense=1, defense=0, staying_power=0)
    m = SalvoModel(ships(), ships())
    with pytest.raises(ValueError):
        m.simulate(-1, 2)
    with pytest.raises(ValueError):
        m.simulate(1, 2, salvos=0)
    with pytest.raises(ValueError):
        m.simulate(1, 2, first="nobody")  # type: ignore[arg-type]


# ------------------------------------------------------------------ mixed forces


def test_single_type_mixed_force_equals_square_law() -> None:
    mf = MixedForceLaw.from_rates({"inf": {"inf": 0.012}}, {"inf": {"inf": 0.009}})
    res = mf.simulate({"inf": 5_000}, {"inf": 6_000}, t_max=20)
    ref = SquareLaw(0.012, 0.009).simulate(Battle(5_000, 6_000), t_max=20)
    assert res.remaining[0]["inf"] == pytest.approx(ref.remaining[0], rel=1e-5)
    assert res.remaining[1]["inf"] == pytest.approx(ref.remaining[1], rel=1e-5)


def test_mixed_force_allocation_and_breakpoint() -> None:
    mf = MixedForceLaw.from_rates(
        {"drone": {"tank": 0.05, "infantry": 0.2}},
        {"tank": {"drone": 0.0}, "infantry": {"drone": 0.01}},
    )
    red_l, blue_l = mf.rates([100], [10, 90])
    # Proportional fire: 10% of drone fire at tanks, 90% at infantry.
    assert blue_l == pytest.approx([100 * 0.05 * 0.1, 100 * 0.2 * 0.9])
    assert red_l == pytest.approx([90 * 0.01])  # tanks can't hit drones
    uniform = MixedForceLaw(mf.red_types, mf.blue_types, mf.red_kills, mf.blue_kills, "uniform")
    assert uniform.rates([100], [10, 90])[1] == pytest.approx([100 * 0.05 * 0.5, 100 * 0.2 * 0.5])
    res = mf.simulate({"drone": 100}, {"tank": 10, "infantry": 90}, t_max=100, breakpoint=0.3)
    assert res.winner == "red"
    assert sum(res.remaining[1].values()) == pytest.approx(70, rel=1e-3)


def test_mixed_force_validation() -> None:
    with pytest.raises(ValueError, match="unknown types"):
        MixedForceLaw.from_rates({"a": {"zzz": 1}}, {"b": {"a": 1}})
    mf = MixedForceLaw.from_rates({"a": {"b": 1}}, {"b": {"a": 1}})
    with pytest.raises(ValueError):
        mf.simulate({"c": 1}, {"b": 1}, t_max=1)
    with pytest.raises(ValueError):
        mf.simulate([1, 2], [1], t_max=1)
    with pytest.raises(ValueError):
        mf.simulate({"a": 1}, {"b": 1}, t_max=-1)
    with pytest.raises(ValueError):
        MixedForceLaw(("a",), ("b",), np.array([[-1.0]]), np.array([[1.0]]))


# ------------------------------------------------------------------ discovery


@pytest.mark.parametrize(("p", "q"), [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.7, 0.3)])
def test_power_law_fit_recovers_exponents(p: float, q: float) -> None:
    rho, beta = 1e-3 / 5_000 ** (p + q - 1), 0.8e-3 / 5_000 ** (p + q - 1)
    runs = []
    for r0, b0 in [(5_000, 6_000), (8_000, 3_000), (2_000, 4_000)]:
        res = GeneralLaw(rho, beta, p=p, q=q).simulate(
            Battle(r0, b0), t_max=200, n_points=201, min_strength=0
        )
        runs.append((res.t, res.red, res.blue))
    fit = fit_power_law(runs)
    for side in (fit.red, fit.blue):
        assert side.p == pytest.approx(p, abs=0.05)
        assert side.q == pytest.approx(q, abs=0.05)
        assert side.r_squared > 0.99
    assert fit.red.exponent == pytest.approx(p - q + 1, abs=0.1)
    assert "enemy^" in str(fit)


def test_power_law_fit_flags_unidentifiable_single_log_law_run() -> None:
    res = GeneralLaw(1e-3, 0.8e-3, p=0, q=1).simulate(Battle(5_000, 6_000), t_max=200, n_points=201)
    with pytest.raises(ValueError, match="lockstep"):
        fit_power_law([(res.t, res.red, res.blue)])


def test_power_law_fit_detects_asymmetry() -> None:
    """Red faces aimed fire, blue faces area fire (Deitchman's guerrilla model)."""
    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        lambda t, y: [-0.01 * y[1], -1e-6 * y[0] * y[1]],
        (0, 40),
        [6_000, 3_000],
        t_eval=np.linspace(0, 40, 81),
        rtol=1e-10,
    )
    fit = fit_power_law([(sol.t, sol.y[0], sol.y[1])])
    assert (fit.red.p, fit.red.q) == pytest.approx((1, 0), abs=0.05)
    assert (fit.blue.p, fit.blue.q) == pytest.approx((1, 1), abs=0.05)


def test_sindy_needs_several_trajectories() -> None:
    def runs(law: SquareLaw | LinearLaw) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        out = []
        for r0, b0 in [(5000, 6000), (8000, 4000), (3000, 3000), (6000, 9000), (4000, 7000)]:
            res = law.simulate(Battle(r0, b0), t_max=30, n_points=121)
            out.append((res.t, res.red, res.blue))
        return out

    sq = sindy(runs(SquareLaw(0.012, 0.009)))
    assert set(sq.red) == {"B"} and sq.red["B"] == pytest.approx(-0.009, rel=0.02)
    assert set(sq.blue) == {"R"} and sq.blue["R"] == pytest.approx(-0.012, rel=0.02)
    lin = sindy(runs(LinearLaw(2e-6, 1.5e-6)))
    assert set(lin.red) == {"R·B"}
    assert lin.red["R·B"] == pytest.approx(-1.5e-6 * lin.scale, rel=0.02)
    # One trajectory conserves rho*R^2 - beta*B^2, so the law is not identifiable.
    single = sindy(runs(SquareLaw(0.012, 0.009))[:1])
    assert set(single.red) != {"B"}
    assert "dR/dt" in str(sq)


def test_discovery_validation() -> None:
    t = np.arange(10.0)
    with pytest.raises(ValueError):
        fit_power_law([(t, np.ones(9), np.ones(10))])
    with pytest.raises(ValueError):
        fit_power_law([(t[::-1], np.ones(10), np.ones(10))])
    with pytest.raises(ValueError):
        fit_power_law([(t, -np.ones(10), np.ones(10))])
    with pytest.raises(ValueError, match="positive losses"):
        fit_power_law([(t, np.ones(10) * 5, np.ones(10) * 5)])
    with pytest.raises(ValueError):
        fit_power_law([])
    with pytest.raises(ValueError):
        sindy([])
    with pytest.raises(ValueError):
        sindy([(t, 10 - t * 0.1, 10 - t * 0.1)], degree=9)
    assert math.isfinite(fit_power_law([(t, 100 - t, 100 - 2 * t**1.5)]).red.p)
