import numpy as np
import pytest

from jomini import Battle, GeneralLaw, LinearLaw, LogarithmicLaw, SquareLaw


@pytest.mark.parametrize("law", [SquareLaw, LinearLaw, LogarithmicLaw])
def test_single_observation_is_reproduced_exactly(law: type) -> None:
    # Cerignola (1503): 6,300 Spanish lost ~500, 9,000 French lost ~4,000.
    battle = Battle(6_300, 9_000)
    model = law.fit(battle, t=1, red=5_800, blue=5_000)
    assert isinstance(model, law)
    remaining = model.simulate(battle, t_max=1).remaining
    assert remaining == pytest.approx((5_800, 5_000), rel=1e-6)


@pytest.mark.parametrize(
    "true",
    [SquareLaw(rho=0.012, beta=0.009), LinearLaw(rho=3e-6, beta=2e-6), LogarithmicLaw(0.05, 0.03)],
)
def test_recovers_coefficients_from_trajectory(true: SquareLaw) -> None:
    battle = Battle(5_000, 6_000)
    ts = np.linspace(1, 15, 8)
    result = true.simulate(battle, t_max=15)
    obs = np.array([result.at(t) for t in ts])
    fitted = type(true).fit(battle, t=ts, red=obs[:, 0], blue=obs[:, 1])
    assert fitted.rho == pytest.approx(true.rho, rel=1e-5)
    assert fitted.beta == pytest.approx(true.beta, rel=1e-5)


def test_general_law_recovers_exponents() -> None:
    true = GeneralLaw(rho=0.04, beta=0.03, p=0.7, q=0.3)
    battle = Battle(5_000, 6_000)
    ts = np.linspace(1, 20, 6)
    result = true.simulate(battle, t_max=20)
    obs = np.array([result.at(t) for t in ts])
    fitted = GeneralLaw.fit(battle, t=ts, red=obs[:, 0], blue=obs[:, 1])
    assert fitted.p == pytest.approx(0.7, abs=1e-4)
    assert fitted.q == pytest.approx(0.3, abs=1e-4)
    assert fitted.rho == pytest.approx(0.04, rel=1e-3)


def test_general_law_with_fixed_exponents() -> None:
    battle = Battle(6_300, 9_000)
    fitted = GeneralLaw.fit(battle, t=1, red=5_800, blue=5_000, p=1, q=0)
    square = SquareLaw.fit(battle, t=1, red=5_800, blue=5_000)
    assert (fitted.p, fitted.q) == (1, 0)
    assert fitted.rho == pytest.approx(square.rho, rel=1e-6)


def test_noisy_data_gives_close_estimate() -> None:
    true = SquareLaw(rho=0.012, beta=0.009)
    battle = Battle(5_000, 6_000)
    ts = np.linspace(1, 15, 30)
    result = true.simulate(battle, t_max=15)
    obs = np.array([result.at(t) for t in ts])
    obs *= np.random.default_rng(0).normal(1, 0.01, obs.shape)
    fitted = SquareLaw.fit(battle, t=ts, red=obs[:, 0], blue=obs[:, 1])
    assert fitted.rho == pytest.approx(true.rho, rel=0.05)
    assert fitted.beta == pytest.approx(true.beta, rel=0.05)


class TestValidation:
    battle = Battle(1_000, 1_000)

    def test_too_little_data_for_exponents(self) -> None:
        with pytest.raises(ValueError, match="observation"):
            GeneralLaw.fit(self.battle, t=1, red=900, blue=800)
        # Fixing one exponent leaves 3 unknowns: still too many for one observation.
        with pytest.raises(ValueError, match="observation"):
            GeneralLaw.fit(self.battle, t=1, red=900, blue=800, p=1)

    @pytest.mark.parametrize(
        ("t", "red", "blue"),
        [
            ([1, 2], [900], [800]),  # length mismatch
            (0, 900, 800),  # t must be positive
            ([2, 1], [900, 800], [800, 700]),  # t must increase
            (1, 1_100, 800),  # more than initial strength
            (1, 900, -5),  # negative
            (1, float("nan"), 800),
        ],
    )
    def test_rejects_bad_observations(self, t: object, red: object, blue: object) -> None:
        with pytest.raises(ValueError):
            SquareLaw.fit(self.battle, t=t, red=red, blue=blue)  # type: ignore[arg-type]

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(TypeError):
            SquareLaw.fit(self.battle, t="soon", red=900, blue=800)  # type: ignore[arg-type]

    def test_rejects_bad_fixed_exponent(self) -> None:
        with pytest.raises(ValueError):
            GeneralLaw.fit(self.battle, t=[1, 2], red=[900, 800], blue=[800, 600], p=-1)
