import numpy as np
import pytest

from jomini import data
from jomini.dynamics import effectiveness_over_time, intensity_over_time


def _synthetic(seed: int = 0, n: int = 150) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    path = np.cumsum(rng.normal(0, 0.08, n)) + 0.3
    blue = rng.poisson(200, n)
    red = rng.poisson(200 * np.exp(-path) * np.exp(rng.normal(0, 0.1, n)))
    return path, red, blue


def test_recovers_hidden_effectiveness_path() -> None:
    truth, red, blue = _synthetic()
    p = effectiveness_over_time(red, blue)
    assert p.step_sd == pytest.approx(0.08, abs=0.03)
    # Overdispersion and random-walk steps are only weakly separable from one
    # series; the path itself is what the method is for.
    assert p.overdispersion_sd < 0.3
    assert np.sqrt(np.mean((p.mean - truth) ** 2)) < 0.1
    lo, hi = p.interval(0.9)
    coverage = np.mean((np.exp(truth) >= lo) & (np.exp(truth) <= hi))
    assert 0.8 <= coverage <= 1.0


def test_exponent_uses_strengths() -> None:
    rl, bl = np.full(10, 100.0), np.full(10, 100.0)
    red, blue = np.full(10, 1000.0), np.full(10, 4000.0)
    linear = effectiveness_over_time(rl, bl, step_sd=0.01, overdispersion_sd=0.01)
    square = effectiveness_over_time(
        rl, bl, red=red, blue=blue, exponent=2.0, step_sd=0.01, overdispersion_sd=0.01
    )
    # Equal losses: linear law says equal effectiveness; square law credits the
    # outnumbered side (red faces 4x the fire yet loses the same).
    assert linear.ratio()[-1] == pytest.approx(1.0, abs=0.01)
    assert square.ratio()[-1] == pytest.approx(4.0, rel=0.02)
    with pytest.raises(ValueError, match="strengths"):
        effectiveness_over_time(rl, bl, exponent=2.0)


def test_intensity_tracks_a_level_shift() -> None:
    rng = np.random.default_rng(1)
    rate = np.r_[np.full(60, 50.0), np.full(60, 200.0)]
    p = intensity_over_time(rng.poisson(rate))
    assert p.ratio()[30] == pytest.approx(50, rel=0.15)
    assert p.ratio()[100] == pytest.approx(200, rel=0.15)
    ratio, sd = p.change(30, 100)
    assert ratio == pytest.approx(4, rel=0.2) and sd > 0


def test_missing_and_negative_values() -> None:
    x = np.array([10, 12, -3, 11, 9, 10, 14, 8], float)  # a downward correction
    p = intensity_over_time(x)
    assert np.isnan(p.observed[2]) and np.isfinite(p.mean).all()


def test_kursk_effectiveness_peaks_at_prokhorovka() -> None:
    k = data.kursk()
    g, s = k.series["german"], k.series["soviet"]
    p = effectiveness_over_time(g["casualties"], s["casualties"])
    assert np.all(p.ratio() > 1)  # Germans more effective per soldier throughout
    assert k.dates[int(np.argmax(p.ratio()))].day == 12


def test_validation() -> None:
    with pytest.raises(ValueError):
        effectiveness_over_time([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        effectiveness_over_time([1, -2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        effectiveness_over_time([1, 2, 3], [1, 2, 3], t=[0, 0, 1])
    with pytest.raises(ValueError):
        intensity_over_time([np.nan, np.nan, 1.0])


def test_loss_series_resample() -> None:
    from datetime import date

    s = data.LossSeries(
        (date(2022, 1, 1), date(2022, 1, 4), date(2022, 1, 10)),
        {"x": np.array([1.0, 5.0, 9.0])},
        "test",
    )
    r = s.resample(3)
    assert [d.day for d in r.dates] == [1, 4, 7, 10]
    assert list(r.cumulative["x"]) == [1, 5, 5, 9]
    assert list(r.daily("x")) == [0, 4, 0, 4]
    with pytest.raises(ValueError):
        s.resample(0)


def test_oryx_events_parsing() -> None:
    text = (
        '"","system","country","origin","status","url","sysID","imageID","statusID","matID","Date","class"\n'
        '"1","T-72","Russia","SU","destroyed","u",1,1,1,"m","02/24/2022","Tanks"\n'
        '"2","T-72","Russia","SU","destroyed","u",1,1,1,"m","02/26/2022","Tanks"\n'
        '"3","BMP","Ukraine","SU","captured","u",1,1,1,"m","02/26/2022","IFVs"\n'
        '"4","x","Russia","SU","destroyed","u",1,1,1,"m","bad","Tanks"\n'
    )
    ru = data.ukraine_oryx_events("Russia", fetch=lambda url: text.encode())
    assert len(ru.dates) == 3
    assert list(ru.cumulative["All Types"]) == [1, 1, 2]
    ua = data.ukraine_oryx_events("Ukraine", fetch=lambda url: text.encode())
    assert ua.dates == ru.dates and list(ua.cumulative["All Types"]) == [0, 0, 1]
