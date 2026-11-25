import math

import numpy as np
import pytest

from jomini import Battle, GeneralLaw, cdb90
from jomini.research import (
    ExponentEstimate,
    effectiveness_table,
    estimate_exponent,
    relative_effectiveness,
)


def synthetic_battles(exponent: float, n: int = 300, seed: int = 1) -> list[tuple[float, ...]]:
    """Battles simulated under GeneralLaw with p - q + 1 == exponent and random quality."""
    rng = np.random.default_rng(seed)
    # e = p - q + 1: square (1, 0), constant-rate (0, 0), logarithmic (0, 1).
    p, q = (exponent - 1.0, 0.0) if exponent >= 1 else (0.0, 1.0 - exponent)
    out = []
    for _ in range(n):
        red, blue = rng.uniform(5_000, 50_000, 2)
        quality = math.exp(rng.normal(0, 0.4))  # independent of force ratio
        base = 0.05 / (red**p * blue**q) * blue  # ~5% of blue lost per unit time
        model = GeneralLaw(rho=base, beta=base / quality, p=p, q=q)
        r1, b1 = model.simulate(Battle(red, blue), t_max=1.0, min_strength=0).remaining
        if r1 > 0 and b1 > 0:
            out.append((red, r1, blue, b1))
    return out


@pytest.mark.parametrize("exponent", [0.0, 1.0, 2.0, 0.5])
def test_recovers_known_exponent(exponent: float) -> None:
    est = estimate_exponent(synthetic_battles(exponent), n_boot=200)
    assert est.exponent == pytest.approx(exponent, abs=0.15)
    assert est.ci_low <= exponent <= est.ci_high


def test_relative_effectiveness_matches_fitted_models() -> None:
    w = cdb90.get("Waterloo")
    for law_e, p, q in ((2, 1, 0), (1, 1, 1)):
        model = w.fit(GeneralLaw, p=p, q=q)
        assert relative_effectiveness(w, law_e) == pytest.approx(model.rho / model.beta, rel=1e-5)


def test_cdb90_rejects_square_law() -> None:
    est = estimate_exponent(cdb90.battles(), n_boot=200)
    assert est.n == 646
    assert est.excludes("square")
    assert 0 < est.exponent < 1.5
    assert "CI" in str(est)


def test_estimate_properties() -> None:
    est = ExponentEstimate(exponent=0.9, ci_low=0.6, ci_high=1.2, confidence=0.95, n=100)
    assert est.nearest_law == "linear"
    assert est.excludes("square") and not est.excludes("linear")


def test_validation() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        estimate_exponent(cdb90.battles()[:5])
    with pytest.raises(ValueError, match="start > end"):
        estimate_exponent([(100, 120, 100, 50)] * 10)
    with pytest.raises(ValueError, match="vary"):
        estimate_exponent([(100, 90, 100, 80)] * 10)
    with pytest.raises(ValueError, match="confidence"):
        estimate_exponent(synthetic_battles(1.0), confidence=1.5)


def test_effectiveness_table() -> None:
    pytest.importorskip("pandas")
    df = effectiveness_table()
    assert len(df) == 646
    assert df.loc[145, "attacker_effectiveness"] == pytest.approx(
        relative_effectiveness(cdb90.get(145), 1.0)
    )
    assert set(df.columns) >= {"force_ratio", "attacker_effectiveness", "winner"}
