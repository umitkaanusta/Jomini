import pytest

from jomini import Battle, GeneralLaw, SquareLaw, cdb90


def test_dataset_shape() -> None:
    battles = cdb90.battles()
    assert len(battles) == 660
    assert len({b.id for b in battles}) == 660
    assert sum(b.fittable for b in battles) == 646
    assert sum(b.narrative is not None for b in battles) == 610
    assert min(b.year for b in battles) == 1600


def test_waterloo_record() -> None:
    w = cdb90.get("Waterloo")
    assert w.id == 145
    assert w.date.isoformat() == "1815-06-18"
    assert w.winner == "defender"
    assert w.attacker.actors == ("France",)
    assert w.attacker.commanders == ("Napoleon I",)
    assert (w.attacker.strength, w.attacker.casualties) == (68_265, 25_000)
    assert w.defender.remaining == 137_547 - 22_500
    assert w.wikipedia_url == "https://en.wikipedia.org/wiki/Battle_of_Waterloo"
    assert w.narrative and "Napoleon" in w.narrative
    assert "~~" not in w.narrative
    assert "Waterloo" in str(w)


@pytest.mark.parametrize("key", [145, "waterloo", "  WATERLOO ", "Battle of Waterloo"])
def test_get_variants(key: int | str) -> None:
    assert cdb90.get(key).id == 145


def test_get_errors() -> None:
    with pytest.raises(KeyError, match="ambiguous"):
        cdb90.get("Kursk")
    with pytest.raises(KeyError, match="Waterloo"):
        cdb90.get("waterlo")
    with pytest.raises(KeyError):
        cdb90.get(99_999)
    with pytest.raises(TypeError):
        cdb90.get(1.5)  # type: ignore[arg-type]


def test_search_filters() -> None:
    napoleon = cdb90.search(commander="napoleon")
    assert napoleon and all(
        any("napoleon" in c.lower() for s in (b.attacker, b.defender) for c in s.commanders)
        for b in napoleon
    )
    ww2 = cdb90.search(war="world war ii", year_from=1944, year_to=1944, winner="attacker")
    assert ww2 and all(b.year == 1944 and b.winner == "attacker" for b in ww2)
    assert {b.id for b in cdb90.search(fittable=False)}.isdisjoint(
        b.id for b in cdb90.search(fittable=True)
    )
    assert cdb90.search(actor="ottoman")
    assert "Gettysburg" in [b.name for b in cdb90.search("gettysburg")]


def test_fit_reproduces_history() -> None:
    w = cdb90.get("Waterloo")
    model = w.fit(SquareLaw)
    remaining = model.simulate(w.to_battle(), t_max=w.duration_days).remaining
    assert remaining == pytest.approx((w.attacker.remaining, w.defender.remaining), rel=1e-6)
    assert isinstance(w.to_battle(), Battle)


def test_fit_passes_kwargs() -> None:
    model = cdb90.get("Gettysburg").fit(GeneralLaw, p=1, q=0)
    assert (model.p, model.q) == (1, 0)


def test_unfittable_battle_raises() -> None:
    bad = cdb90.search(fittable=False)[0]
    with pytest.raises(ValueError, match="fit"):
        bad.fit(SquareLaw)


def test_to_pandas() -> None:
    pytest.importorskip("pandas")
    df = cdb90.to_pandas()
    assert df.shape[0] == 660
    assert df.loc[145, "attacker_strength"] == 68_265
    assert df.loc[145, "attacker_actors"] == "France"
    assert int(df["fittable"].sum()) == 646
