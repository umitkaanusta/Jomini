import json
from pathlib import Path

import numpy as np
import pytest

from jomini import cdb90, data
from jomini.discovery import fit_power_law_from_losses
from jomini.research import estimate_exponent


def test_kursk_matches_published_tables() -> None:
    k = data.kursk()
    assert len(k.dates) == 14 and k.dates[0].isoformat() == "1943-07-05"
    sov, ger = k.series["soviet"], k.series["german"]
    assert sov["manpower"][0] == 507_698 and ger["casualties"][0] == 6_192
    prokhorovka = k.dates.index(next(d for d in k.dates if d.day == 12))
    assert sov["tanks_lost"][prokhorovka] == 414  # Lucas & Turkes, Table 3 note
    # German tanks on hand match Turkes (2000) Table 11 day by day.
    assert list(ger["tanks"]) == [
        986,
        749,
        673,
        596,
        490,
        548,
        563,
        500,
        495,
        480,
        426,
        495,
        557,
        588,
    ]
    assert k.combat_power("german")[0] == 301_341 + 5 * 1142 + 20 * 986 + 40 * 1166
    t, red, blue = k.trajectory("manpower")
    assert t[-1] == 13 and red[0] == 301_341 and blue[0] == 507_698
    with pytest.raises(ValueError):
        k.trajectory("morale")


def test_kursk_exponents_are_not_identified() -> None:
    """Reproduces the literature: 14 days can't pin down Lanchester exponents."""
    k = data.kursk()
    g, s = k.series["german"], k.series["soviet"]
    fit = fit_power_law_from_losses(g["manpower"], s["manpower"], g["casualties"], s["casualties"])
    assert fit.red.p > 3 and fit.red.q_se > 3  # implausible exponent, huge uncertainty


def test_kursk_to_pandas() -> None:
    pytest.importorskip("pandas")
    df = data.kursk().to_pandas()
    assert df.shape[0] == 28 and set(df.side) == {"german", "soviet"}


def test_livermore() -> None:
    battles = data.livermore()
    assert len(battles) == 63 and sum(b.fittable for b in battles) == 48
    g = next(b for b in battles if b.name == "Gettysburg")
    cg = cdb90.get("Gettysburg")
    assert (g.attacker.strength, g.defender.strength) == (
        cg.attacker.strength,
        cg.defender.strength,
    )
    assert g.winner == "defender" and g.duration_days == 3
    assert estimate_exponent(battles, n_boot=0).n == 48


def _fake_fetch(responses: dict[str, bytes]):  # type: ignore[no-untyped-def]
    def fetch(url: str) -> bytes:
        return responses[url.rsplit("/", 1)[-1]]

    return fetch


def test_ukraine_general_staff_parsing() -> None:
    personnel = [
        {"date": "2022-02-25", "day": 2, "personnel": 2800, "personnel*": "about", "POW": 0},
        {"date": "2022-02-26", "day": 3, "personnel": 3500, "personnel*": "about", "POW": 0},
    ]
    equipment = [
        {"date": "2022-02-25", "day": 2, "tank": 80, "APC": 516},
        {"date": "2022-02-26", "day": 3, "tank": 102, "APC": 536, "drone": 2},
    ]
    gs = data.ukraine_general_staff(
        fetch=_fake_fetch(
            {
                "russia_losses_personnel.json": json.dumps(personnel).encode(),
                "russia_losses_equipment.json": json.dumps(equipment).encode(),
            }
        )
    )
    assert [d.isoformat() for d in gs.dates] == ["2022-02-25", "2022-02-26"]
    assert list(gs.cumulative["personnel"]) == [2800, 3500]
    assert list(gs.daily("tank")) == [0, 22]
    assert list(gs.cumulative["drone"]) == [0, 2]  # missing early values filled with 0
    assert "POW" in gs.cumulative and "day" not in gs.cumulative


def test_ukraine_oryx_parsing() -> None:
    csv_text = (
        "country,equipment_type,destroyed,abandoned,captured,damaged,type_total,date_recorded\n"
        "Russia,Tanks,10,1,2,0,13,2022-03-01\n"
        "Russia,Tanks,20,1,2,0,23,2022-03-08\n"
        "Russia,Aircraft,3,0,0,0,3,2022-03-08\n"
        "Ukraine,Tanks,5,0,1,0,6,2022-03-01\n"
    )
    ru = data.ukraine_oryx("Russia", fetch=_fake_fetch({"daily_count.csv": csv_text.encode()}))
    assert list(ru.cumulative["Tanks"]) == [13, 23]
    assert list(ru.cumulative["Aircraft"]) == [0, 3]
    assert list(ru.cumulative["All Types"]) == [13, 26]
    with pytest.raises(ValueError):
        data.ukraine_oryx("Belarus")  # type: ignore[arg-type]


def test_fetch_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_get(url: str) -> bytes:
        calls.append(url)
        return b"payload"

    monkeypatch.setenv("JOMINI_CACHE", str(tmp_path))
    monkeypatch.setattr(data, "_http_get", fake_get)
    assert data._fetch("https://x/y/file.json", 1, None) == b"payload"
    assert data._fetch("https://x/y/file.json", 1, None) == b"payload"
    assert len(calls) == 1 and (tmp_path / "file.json").exists()
    data._fetch("https://x/y/file.json", 0, None)  # expired: refetch
    assert len(calls) == 2


def test_losses_fit_validation() -> None:
    with pytest.raises(ValueError):
        fit_power_law_from_losses([1, 2], [1, 2], [0, 0], [0, 0])
    with pytest.raises(ValueError):
        fit_power_law_from_losses(np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
