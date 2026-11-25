import io
import zipfile

import numpy as np
import pytest

from jomini import data
from jomini.escalation import fit_hawkes, simulate_hawkes


def test_recovers_branching_ratio_with_constant_baseline() -> None:
    y = simulate_hawkes(1500, baseline_rate=4, branching_ratio=0.6, decay=0.8, dispersion=5, seed=1)
    fit = fit_hawkes(y, baseline="constant")
    assert fit.branching_ratio == pytest.approx(0.6, abs=0.08)
    assert fit.decay == pytest.approx(0.8, abs=0.08)
    assert fit.p_value < 1e-6 and fit.aic < 2 * 3 - 2 * fit.null_log_likelihood


def test_spline_baseline_is_conservative_but_detects_excitation() -> None:
    y = simulate_hawkes(1500, baseline_rate=4, branching_ratio=0.6, decay=0.8, dispersion=5, seed=1)
    fit = fit_hawkes(y)
    assert 0.3 < fit.branching_ratio <= 0.65 and fit.p_value < 1e-6


def test_no_false_contagion_when_intensity_drifts() -> None:
    """Slow exogenous drift fools a constant baseline but not the spline baseline."""
    rng = np.random.default_rng(2)
    level = 5 * np.exp(np.sin(np.arange(1400) / 120))
    y = rng.negative_binomial(5, 5 / (5 + level))
    assert fit_hawkes(y, baseline="constant").branching_ratio > 0.8  # spurious
    honest = fit_hawkes(y, baseline="spline")
    assert honest.branching_ratio < 0.1 and honest.p_value > 0.05


def test_trend_baseline_and_report() -> None:
    y = simulate_hawkes(800, baseline_rate=3, branching_ratio=0.3, decay=0.5, seed=4)
    fit = fit_hawkes(y, baseline="trend")
    assert fit.trend_per_year > 0 and "branching ratio" in str(fit)
    assert fit.half_life_days > 0 and fit.baseline_path.size == 800


def test_validation() -> None:
    with pytest.raises(ValueError):
        fit_hawkes(np.ones(20))
    with pytest.raises(ValueError):
        fit_hawkes(-np.ones(200))
    with pytest.raises(ValueError):
        fit_hawkes(np.zeros(200))
    with pytest.raises(ValueError):
        fit_hawkes(np.ones(200), baseline="wiggly")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        simulate_hawkes(10, baseline_rate=1, branching_ratio=1.2, decay=0.5)


GED_HEADER = (
    "id,date_start,date_end,conflict_name,dyad_name,side_a,side_b,country,deaths_a,deaths_b,"
    "deaths_civilians,deaths_unknown,best,low,high,latitude,longitude,type_of_violence,date_prec"
)


def _ged_zip(rows: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("GEDEvent_v26_1.csv", "\n".join([GED_HEADER, *rows]) + "\n")
    return buf.getvalue()


def test_ged_loader_select_and_daily() -> None:
    rows = [
        "1,2022-03-01 00:00:00.000,2022-03-01 00:00:00.000,Russia - Ukraine,Russia - Ukraine,"
        "Russia,Ukraine,Ukraine,10,4,1,0,15,12,20,48.0,37.0,1,1",
        "2,2022-03-01 00:00:00.000,2022-03-03 00:00:00.000,Russia - Ukraine,Russia - Ukraine,"
        "Russia,Ukraine,Ukraine,30,0,0,0,30,30,30,48.0,37.0,1,4",
        "3,2022-03-02 00:00:00.000,2022-03-02 00:00:00.000,Sudan: Government,SAF - RSF,"
        "SAF,RSF,Sudan,1,1,0,0,2,2,2,15.0,32.0,1,1",
    ]
    ged = data.ucdp_ged(fetch=lambda url: _ged_zip(rows))
    assert len(ged) == 3
    ru = ged.select(conflict="russia")
    assert len(ru) == 2 and len(ged.select(country="Sudan")) == 1
    deaths = ru.daily("deaths_a")
    assert [d.day for d in deaths.dates] == [1, 2, 3]
    assert list(deaths.daily("deaths_a")) == [20.0, 10.0, 10.0]  # 10 + 30 spread over 3 days
    assert list(ru.daily("events").cumulative["events"]) == pytest.approx([4 / 3, 5 / 3, 2])
    with pytest.raises(ValueError):
        ru.daily("morale")
    with pytest.raises(ValueError):
        ged.select(country="Atlantis").daily()


def test_candidate_loader_reads_plain_csv() -> None:
    rows = [
        "1,2026-02-28 00:00:00.000,2026-02-28 00:00:00.000,Iran - Israel,Government of Iran - "
        "Government of Israel,Iran,Israel,Iran,50,0,20,0,70,60,90,35.7,51.4,1,1",
    ]
    csv_bytes = ("\n".join([GED_HEADER, *rows]) + "\n").encode()
    cand = data.ucdp_candidate(fetch=lambda url: csv_bytes)
    assert len(cand) == 1 and cand.select(conflict="iran")["best"][0] == 70
