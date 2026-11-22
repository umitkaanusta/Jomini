import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jomini import (
    Battle,
    GeneralLaw,
    LinearLaw,
    LogarithmicLaw,
    Outcome,
    SquareLaw,
)


def square_law_exact(
    r0: float, b0: float, rho: float, beta: float, t: float
) -> tuple[float, float]:
    k = math.sqrt(rho * beta)
    red = r0 * math.cosh(k * t) - math.sqrt(beta / rho) * b0 * math.sinh(k * t)
    blue = b0 * math.cosh(k * t) - math.sqrt(rho / beta) * r0 * math.sinh(k * t)
    return red, blue


class TestSquareLaw:
    def test_matches_closed_form(self) -> None:
        battle = Battle(20_000, 30_000)
        result = SquareLaw(rho=0.015, beta=0.012).simulate(battle, t_max=40)
        for t in (0.0, 5.0, 17.3, 40.0):
            expected = square_law_exact(20_000, 30_000, 0.015, 0.012, t)
            assert result.at(t) == pytest.approx(expected, rel=1e-7)

    def test_final_strength_from_invariant(self) -> None:
        result = SquareLaw(rho=0.015, beta=0.012).simulate(Battle(20_000, 30_000), min_strength=0)
        assert result.winner == "blue"
        assert result.outcome is Outcome.ANNIHILATION
        expected_blue = math.sqrt((0.012 * 30_000**2 - 0.015 * 20_000**2) / 0.012)
        assert result.remaining[1] == pytest.approx(expected_blue, rel=1e-7)
        assert result.remaining[0] == pytest.approx(0, abs=1e-3)

    def test_concentration_of_force(self) -> None:
        # Equal quality, 2:1 numbers: the larger side loses only ~13% (1 - sqrt(3)/2).
        result = SquareLaw(0.01, 0.01).simulate(Battle(2_000, 1_000), min_strength=0)
        assert result.winner == "red"
        assert result.casualties[0] / 2_000 == pytest.approx(1 - math.sqrt(3) / 2, rel=1e-6)


class TestLinearLaw:
    def test_conserves_difference(self) -> None:
        model = LinearLaw(rho=2e-6, beta=1e-6)
        result = model.simulate(Battle(10_000, 15_000))
        r, b = result.red, result.blue
        np.testing.assert_allclose(2e-6 * r - 1e-6 * b, 2e-6 * 10_000 - 1e-6 * 15_000, rtol=1e-6)
        assert result.winner == "red"


class TestLogarithmicLaw:
    def test_each_side_decays_exponentially(self) -> None:
        # With q=1, p=0: dR/dt = -beta R, so R(t) = R0 exp(-beta t).
        result = LogarithmicLaw(rho=0.02, beta=0.01).simulate(Battle(5_000, 5_000), t_max=50)
        assert result.outcome is Outcome.TIME_LIMIT
        assert result.winner is None
        red, blue = result.at(50)
        assert red == pytest.approx(5_000 * math.exp(-0.5), rel=1e-7)
        assert blue == pytest.approx(5_000 * math.exp(-1.0), rel=1e-7)


class TestGeneralLaw:
    @pytest.mark.parametrize(
        ("special", "p", "q"),
        [(SquareLaw, 1, 0), (LinearLaw, 1, 1), (LogarithmicLaw, 0, 1)],
    )
    def test_reduces_to_special_cases(self, special: type, p: float, q: float) -> None:
        battle = Battle(1_000, 1_200)
        a = special(rho=1e-3, beta=8e-4).simulate(battle, t_max=30)
        b = GeneralLaw(rho=1e-3, beta=8e-4, p=p, q=q).simulate(battle, t_max=30)
        np.testing.assert_allclose(a.red, b.red)
        np.testing.assert_allclose(a.blue, b.blue)

    @pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
    def test_rejects_bad_exponents(self, bad: float) -> None:
        with pytest.raises(ValueError):
            GeneralLaw(0.01, 0.01, p=bad)


class TestBreakpoint:
    def test_side_breaks_at_threshold(self) -> None:
        result = SquareLaw(0.01, 0.01).simulate(Battle(10_000, 12_000), breakpoint=0.3)
        assert result.outcome is Outcome.BREAKPOINT
        assert result.loser == "red"
        assert result.remaining[0] == pytest.approx(7_000, rel=1e-8)

    def test_per_side_breakpoint(self) -> None:
        # Red is outnumbered but far more stubborn: blue breaks first.
        result = SquareLaw(0.01, 0.01).simulate(Battle(10_000, 12_000), breakpoint=(0.9, 0.1))
        assert result.winner == "red"
        assert result.remaining[1] == pytest.approx(10_800, rel=1e-8)

    @pytest.mark.parametrize("bad", [0, -0.5, 1.5, (0.2,), (0.2, 0.3, 0.4)])
    def test_rejects_bad_breakpoint(self, bad: object) -> None:
        with pytest.raises((ValueError, TypeError)):
            SquareLaw(0.01, 0.01).simulate(Battle(100, 100), breakpoint=bad)  # type: ignore[arg-type]


class TestValidation:
    @pytest.mark.parametrize(("red", "blue"), [(0, 10), (10, -1), (math.inf, 10), (10, math.nan)])
    def test_battle_rejects_bad_strengths(self, red: float, blue: float) -> None:
        with pytest.raises(ValueError):
            Battle(red, blue)

    @pytest.mark.parametrize("bad", ["10", None, True])
    def test_battle_rejects_non_numbers(self, bad: object) -> None:
        with pytest.raises(TypeError):
            Battle(bad, 10)  # type: ignore[arg-type]

    @pytest.mark.parametrize(("rho", "beta"), [(0, 0.1), (0.1, -1), (math.nan, 0.1)])
    def test_model_rejects_bad_coefficients(self, rho: float, beta: float) -> None:
        with pytest.raises(ValueError):
            SquareLaw(rho, beta)

    def test_simulate_requires_battle(self) -> None:
        with pytest.raises(TypeError):
            SquareLaw(0.1, 0.1).simulate((100, 100))  # type: ignore[arg-type]

    def test_min_strength_below_initial(self) -> None:
        with pytest.raises(ValueError):
            SquareLaw(0.1, 0.1).simulate(Battle(0.5, 0.8))
        # Normalised forces work once min_strength is scaled accordingly.
        result = SquareLaw(0.1, 0.1).simulate(Battle(0.5, 0.8), min_strength=1e-3)
        assert result.winner == "blue"

    def test_linear_law_needs_floor(self) -> None:
        # Linear law with equal "strength" only decays hyperbolically: never reaches 0.
        with pytest.raises(RuntimeError):
            LinearLaw(1e-3, 1e-3).simulate(Battle(100, 100), min_strength=0)


class TestResult:
    def test_trajectory_shape_and_bounds(self) -> None:
        result = SquareLaw(0.015, 0.012).simulate(Battle(20_000, 30_000), n_points=50)
        assert result.t.shape == result.red.shape == result.blue.shape == (50,)
        assert result.t[0] == 0 and result.t[-1] == pytest.approx(result.duration)
        assert result.initial == (20_000, 30_000)
        assert "blue wins" in str(result)

    def test_at_clamps_to_duration(self) -> None:
        result = SquareLaw(0.015, 0.012).simulate(Battle(20_000, 30_000))
        assert result.at(1e9) == result.at(result.duration)
        with pytest.raises(ValueError):
            result.at(-1)

    def test_to_pandas(self) -> None:
        pytest.importorskip("pandas")
        df = SquareLaw(0.015, 0.012).simulate(Battle(20_000, 30_000), n_points=10).to_pandas()
        assert list(df.columns) == ["red", "blue"] and len(df) == 10

    def test_plot(self) -> None:
        mpl = pytest.importorskip("matplotlib")
        mpl.use("Agg")
        ax = SquareLaw(0.015, 0.012).simulate(Battle(20_000, 30_000)).plot()
        assert len(ax.lines) == 2


strength = st.floats(min_value=10, max_value=1e6)
coef = st.floats(min_value=1e-4, max_value=1e-1)
exponent = st.floats(min_value=0, max_value=1)


@settings(max_examples=60, deadline=None)
@given(r=strength, b=strength, rho=coef, beta=coef, p=exponent, q=exponent)
def test_strengths_monotone_and_non_negative(
    r: float, b: float, rho: float, beta: float, p: float, q: float
) -> None:
    result = GeneralLaw(rho, beta, p=p, q=q).simulate(Battle(r, b), t_max=1_000, breakpoint=0.5)
    for series in (result.red, result.blue):
        assert np.all(series >= 0)
        assert np.all(np.diff(series) <= 1e-6 * series[0])


@settings(max_examples=60, deadline=None)
@given(r=strength, b=strength, rho=coef, beta=coef)
def test_square_law_invariant_holds(r: float, b: float, rho: float, beta: float) -> None:
    result = SquareLaw(rho, beta).simulate(Battle(r, b))
    invariant = rho * result.red**2 - beta * result.blue**2
    np.testing.assert_allclose(
        invariant, rho * r**2 - beta * b**2, rtol=1e-5, atol=1e-6 * max(r, b) ** 2 * max(rho, beta)
    )
