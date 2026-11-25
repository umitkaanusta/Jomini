import math

import numpy as np
import pytest

from jomini import Battle, LinearLaw, SquareLaw, cdb90
from jomini.stochastic import BreakpointModel, MonteCarloResult, default_breakpoints, monte_carlo


class _Side:
    def __init__(self, frac: float) -> None:
        self.loss_fraction = frac


class _Battle:
    def __init__(self, year: int, att: float, dfn: float, winner: str) -> None:
        self.year, self.winner = year, winner
        self.attacker, self.defender = _Side(att), _Side(dfn)


def test_breakpoint_fit_recovers_censored_truth() -> None:
    """Simulate battles where the first side to hit its threshold loses."""
    truth = BreakpointModel(intercept=-1.5, year_slope=-0.4, defender_shift=0.3, sigma=0.6)
    rng = np.random.default_rng(0)
    battles = []
    for _ in range(3000):
        year = int(rng.uniform(1600, 1980))
        ta = truth.sample(rng, 1, year, "attacker")[0]
        td = truth.sample(rng, 1, year, "defender")[0]
        # Both sides lose at random relative speeds; whoever reaches threshold first breaks.
        speed_a, speed_d = rng.lognormal(0, 0.5, 2)
        t_a, t_d = ta / speed_a, td / speed_d
        t = min(t_a, t_d)
        att, dfn = min(speed_a * t, 0.999), min(speed_d * t, 0.999)
        battles.append(_Battle(year, att, dfn, "defender" if t_a < t_d else "attacker"))
    fit = BreakpointModel.fit(battles)  # type: ignore[arg-type]
    assert fit.intercept == pytest.approx(truth.intercept, abs=0.1)
    assert fit.year_slope == pytest.approx(truth.year_slope, abs=0.1)
    assert fit.defender_shift == pytest.approx(truth.defender_shift, abs=0.1)
    assert fit.sigma == pytest.approx(truth.sigma, abs=0.07)


def test_default_breakpoints_on_cdb90() -> None:
    bp = default_breakpoints()
    assert bp is default_breakpoints()
    assert bp.n == 1300 and bp.sigma > 0
    # Armies broke at lower losses as warfare modernised.
    assert bp.median(1700, "attacker") > bp.median(1800, "attacker") > bp.median(1950, "attacker")
    assert bp.median(1900, "defender") > bp.median(1900, "attacker")
    assert 0.1 < bp.median(1900, "attacker") < 0.3
    assert bp.quantile(0.1, 1900, "attacker") < bp.median(1900, "attacker")


def test_fixed_breakpoints_match_deterministic_simulation() -> None:
    model, battle = SquareLaw(0.01, 0.01), Battle(10_000, 12_000)
    det = model.simulate(battle, breakpoint=(0.3, 0.4))
    mc = monte_carlo(model, battle, n=5, breakpoints=(0.3, 0.4), seed=0)
    assert set(mc.winners) == {"blue"} and mc.p_blue == 1
    assert mc.durations[0] == pytest.approx(det.duration, rel=2e-3)
    assert mc.red_casualties[0] == pytest.approx(det.casualties[0], rel=2e-3)
    assert mc.blue_casualties[0] == pytest.approx(det.casualties[1], rel=2e-3)


def test_symmetric_battle_is_a_coin_flip() -> None:
    r = monte_carlo(
        LinearLaw(1e-5, 1e-5),
        Battle(10_000, 10_000),
        n=4000,
        red_role="attacker",
        breakpoints=BreakpointModel(-1.6, 0.0, 0.0, 0.8),
        seed=1,
    )
    assert r.p_red + r.p_blue + r.p_undecided == pytest.approx(1)
    assert r.p_red == pytest.approx(0.5, abs=0.03)


def test_more_troops_raise_win_probability() -> None:
    w = cdb90.get("Waterloo")
    model = w.fit(LinearLaw)
    base = monte_carlo(model, w.to_battle(), year=w.year, seed=0)
    more = monte_carlo(
        model,
        Battle(w.attacker.strength * 1.5, w.defender.strength),
        year=w.year,
        seed=0,  # type: ignore[operator]
    )
    assert 0 < base.p_red < more.p_red < 1


def test_effectiveness_noise_spreads_outcomes() -> None:
    battle, model = Battle(10_000, 10_000), LinearLaw(1e-5, 2e-5)
    fixed = monte_carlo(model, battle, breakpoints=(0.3, 0.3), seed=0, n=500)
    noisy = monte_carlo(model, battle, breakpoints=(0.3, 0.3), effectiveness_sd=1.0, seed=0, n=500)
    assert fixed.p_blue == 1
    assert 0.5 < noisy.p_blue < 1


def test_time_limit_leaves_runs_undecided() -> None:
    r = monte_carlo(LinearLaw(1e-7, 1e-7), Battle(10_000, 10_000), t_max=0.01, seed=0, n=100)
    assert r.p_undecided == 1
    assert np.all(r.durations == pytest.approx(0.01))


def test_validation() -> None:
    m, b = LinearLaw(1e-5, 1e-5), Battle(100, 100)
    for kwargs in (
        {"n": 0},
        {"effectiveness_sd": -1},
        {"t_max": -1},
        {"min_strength": 500},
        {"breakpoints": (0.0, 0.5)},
        {"red_role": "neutral"},
    ):
        with pytest.raises(ValueError):
            monte_carlo(m, b, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        monte_carlo("model", b)  # type: ignore[arg-type]


def test_result_summary() -> None:
    r = MonteCarloResult(
        winners=np.array(["red", "blue", "red", ""]),
        durations=np.array([1.0, 2.0, 3.0, 4.0]),
        red_casualties=np.array([10.0, 20.0, 30.0, 40.0]),
        blue_casualties=np.array([5.0, 5.0, 5.0, 5.0]),
    )
    assert (r.n, r.p_red, r.p_blue, r.p_undecided) == (4, 0.5, 0.25, 0.25)
    assert "red wins 50.0%" in str(r) and "undecided" in str(r)
    assert math.isclose(r.quantiles(r.durations, (0.5,))[0], 2.5)
