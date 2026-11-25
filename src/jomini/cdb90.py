"""CDB90: the US Army Concepts Analysis Agency database of 660 battles (1600-1982).

The data is the open, cleaned-up edition maintained at
https://github.com/jrnold/CDB90 (CC0), bundled with Jomini. By convention the
**attacker is red** and the **defender is blue**.

>>> from jomini import cdb90, SquareLaw
>>> waterloo = cdb90.get("Waterloo")
>>> waterloo.winner
'defender'
>>> model = waterloo.fit(SquareLaw)
"""

from __future__ import annotations

import csv
import difflib
import gzip
import io
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields
from datetime import date
from functools import cache
from importlib import resources
from typing import Any, Literal, TypeVar

from jomini.battle import Battle
from jomini.models import LanchesterModel

VERSION = "10.8.0"
"""CDB90 release the bundled data was built from."""

Role = Literal["attacker", "defender"]
Winner = Literal["attacker", "defender", "draw"]
M = TypeVar("M", bound=LanchesterModel)


@dataclass(frozen=True, slots=True)
class Side:
    """One side of a historical battle. Counts are ``None`` where CDB90 has no value."""

    force: str
    """Unit or army, e.g. ``'CS Army of Northern Virginia'``."""
    actors: tuple[str, ...]
    """States or factions, e.g. ``('Confederacy',)``."""
    commanders: tuple[str, ...]
    strength: int | None
    casualties: int | None
    cavalry: int | None
    tanks: int | None
    artillery: int | None
    aircraft: int | None

    @property
    def remaining(self) -> int | None:
        if self.strength is None or self.casualties is None:
            return None
        return self.strength - self.casualties

    @property
    def loss_fraction(self) -> float | None:
        if self.strength is None or self.casualties is None:
            return None
        return self.casualties / self.strength


@dataclass(frozen=True, slots=True)
class Advantages:
    """Analysts' ratings of the **attacker's** relative advantage in each factor.

    Integers, usually -1 to +1 (some run -3 to +3): positive favours the attacker,
    negative the defender, 0 means neither, ``None`` means not rated. They were
    assigned by HERO analysts *after* the battles, with the outcome known.
    """

    surprise: int | None
    air_superiority: int | None
    leadership: int | None
    training: int | None
    morale: int | None
    logistics: int | None
    momentum: int | None
    intelligence: int | None
    technology: int | None
    initiative: int | None

    def as_dict(self) -> dict[str, int | None]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @property
    def complete(self) -> bool:
        return all(v is not None for v in self.as_dict().values())


ADVANTAGE_FACTORS: tuple[str, ...] = tuple(f.name for f in fields(Advantages))
Posture = Literal["hasty", "prepared", "fortified", "delay", "withdrawal"]


@dataclass(frozen=True, slots=True)
class HistoricalBattle:
    """A battle from CDB90."""

    id: int
    """CDB90 sequence number (``isqno``)."""
    name: str
    war: str
    theater: str | None
    campaign: str | None
    location: str | None
    start: str
    """Start as an ISO 8601 datetime string."""
    end: str
    duration_days: float
    """Days of *active* combat (not calendar days)."""
    latitude: float | None
    longitude: float | None
    front_width_km: float | None
    winner: Winner | None
    attacker: Side
    defender: Side
    wikipedia_title: str | None
    wikidata_id: str | None
    defender_posture: Posture | None
    """Defender's primary posture: hasty, prepared or fortified defence, delay, withdrawal."""
    advantages: Advantages

    @property
    def year(self) -> int:
        return int(self.start[:4])

    @property
    def date(self) -> date:
        return date.fromisoformat(self.start[:10])

    @property
    def wikipedia_url(self) -> str | None:
        if not self.wikipedia_title:
            return None
        return "https://en.wikipedia.org/wiki/" + self.wikipedia_title.replace(" ", "_")

    @property
    def narrative(self) -> str | None:
        """Account of the battle from the HERO study behind CDB90 (610 of 660 battles)."""
        entry = _narratives().get(self.id)
        return entry[0] if entry else None

    @property
    def significance(self) -> str | None:
        entry = _narratives().get(self.id)
        return entry[1] if entry and entry[1] else None

    @property
    def fittable(self) -> bool:
        """Whether strengths, casualties and duration are all present and consistent."""
        a, d = self.attacker, self.defender
        return (
            self.duration_days > 0
            and None not in (a.strength, a.casualties, d.strength, d.casualties)
            and 0 < a.casualties < a.strength  # type: ignore[operator]
            and 0 < d.casualties < d.strength  # type: ignore[operator]
        )

    def to_battle(self) -> Battle:
        """Starting strengths as a :class:`~jomini.Battle` (red = attacker, blue = defender)."""
        if self.attacker.strength is None or self.defender.strength is None:
            raise ValueError(f"{self.name}: CDB90 has no strength for one of the sides")
        return Battle(red=self.attacker.strength, blue=self.defender.strength)

    def fit(self, law: type[M], **kwargs: Any) -> M:
        """Fit ``law`` so that it reproduces this battle's losses over its active duration.

        Time is measured in days, so ``rho`` and ``beta`` are per-day coefficients.
        Extra keyword arguments go to ``law.fit`` (e.g. ``p``/``q`` for GeneralLaw).
        """
        if not self.fittable:
            raise ValueError(
                f"{self.name}: needs both sides' strength and casualties (0 < casualties < "
                "strength) and a positive duration to fit"
            )
        red, blue = self.attacker.remaining, self.defender.remaining
        assert red is not None and blue is not None  # guaranteed by fittable
        return law.fit(self.to_battle(), t=self.duration_days, red=red, blue=blue, **kwargs)

    def side(self, role: Role) -> Side:
        return self.attacker if role == "attacker" else self.defender

    def __str__(self) -> str:
        a, d = self.attacker, self.defender

        def fmt(n: int | None) -> str:
            return "?" if n is None else f"{n:,}"

        return (
            f"{self.name} ({self.date.isoformat()}, {self.war})\n"
            f"Attacker: {', '.join(a.actors) or a.force}: "
            f"{fmt(a.strength)} men, {fmt(a.casualties)} casualties\n"
            f"Defender: {', '.join(d.actors) or d.force}: "
            f"{fmt(d.strength)} men, {fmt(d.casualties)} casualties\n"
            f"Winner: {self.winner or 'unknown'}"
        )


# --------------------------------------------------------------------------- loading


def _open(name: str) -> Iterator[dict[str, str]]:
    raw = resources.files("jomini.datasets").joinpath(name).read_bytes()
    text = gzip.decompress(raw).decode("utf-8")
    yield from csv.DictReader(io.StringIO(text))


def _str(value: str) -> str | None:
    return value or None


def _int(value: str) -> int | None:
    return int(value) if value else None


def _float(value: str) -> float | None:
    return float(value) if value else None


def _list(value: str, sep: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(sep) if part.strip())


def _side(row: dict[str, str], role: str) -> Side:
    return Side(
        force=row[f"{role}_force"],
        actors=_list(row[f"{role}_actors"], "&"),
        commanders=_list(row[f"{role}_commanders"], ";"),
        strength=_int(row[f"{role}_strength"]),
        casualties=_int(row[f"{role}_casualties"]),
        cavalry=_int(row[f"{role}_cavalry"]),
        tanks=_int(row[f"{role}_tanks"]),
        artillery=_int(row[f"{role}_artillery"]),
        aircraft=_int(row[f"{role}_aircraft"]),
    )


@cache
def battles() -> tuple[HistoricalBattle, ...]:
    """All 660 battles, in CDB90 order (roughly chronological)."""
    out = []
    for row in _open("cdb90.csv.gz"):
        out.append(
            HistoricalBattle(
                id=int(row["id"]),
                name=row["name"],
                war=row["war"],
                theater=_str(row["theater"]),
                campaign=_str(row["campaign"]),
                location=_str(row["location"]),
                start=row["start"],
                end=row["end"],
                duration_days=float(row["duration_days"]),
                latitude=_float(row["latitude"]),
                longitude=_float(row["longitude"]),
                front_width_km=_float(row["front_width_km"]),
                winner=_str(row["winner"]),  # type: ignore[arg-type]
                attacker=_side(row, "attacker"),
                defender=_side(row, "defender"),
                wikipedia_title=_str(row["wikipedia_title"]),
                wikidata_id=_str(row["wikidata_id"]),
                defender_posture=_str(row["defender_posture"]),  # type: ignore[arg-type]
                advantages=Advantages(
                    **{name: _int(row[f"adv_{name}"]) for name in ADVANTAGE_FACTORS}
                ),
            )
        )
    return tuple(out)


@cache
def _narratives() -> dict[int, tuple[str, str]]:
    return {
        int(r["id"]): (r["narrative"], r["significance"]) for r in _open("cdb90_narratives.csv.gz")
    }


@cache
def _by_id() -> dict[int, HistoricalBattle]:
    return {b.id: b for b in battles()}


# --------------------------------------------------------------------------- lookup


def get(key: int | str) -> HistoricalBattle:
    """Look a battle up by CDB90 id, name or Wikipedia title (case-insensitive).

    Several CDB90 entries can share a Wikipedia article (Kursk is split into
    phases), so a title only matches when it is unambiguous. Raises ``KeyError``
    with suggestions when nothing matches.
    """
    if isinstance(key, int) and not isinstance(key, bool):
        try:
            return _by_id()[key]
        except KeyError:
            raise KeyError(f"no CDB90 battle with id {key}") from None
    if not isinstance(key, str):
        raise TypeError("key must be an int id or a str name")

    needle = key.strip().casefold()
    by_name = [b for b in battles() if b.name.casefold() == needle]
    if len(by_name) == 1:
        return by_name[0]
    by_title = [
        b
        for b in battles()
        if b.wikipedia_title
        and needle
        in (b.wikipedia_title.casefold(), b.wikipedia_title.casefold().removeprefix("battle of "))
    ]
    candidates = by_name or by_title
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        options = ", ".join(f"{b.name} (id {b.id})" for b in candidates[:8])
        raise KeyError(f"{key!r} is ambiguous: {options}")
    names = [b.name for b in battles()]
    close = difflib.get_close_matches(key.title(), names, n=5, cutoff=0.6)
    hint = f"; did you mean {', '.join(close)}?" if close else "; try cdb90.search()"
    raise KeyError(f"no CDB90 battle named {key!r}{hint}")


def search(
    text: str | None = None,
    *,
    war: str | None = None,
    actor: str | None = None,
    commander: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    winner: Winner | None = None,
    fittable: bool | None = None,
) -> list[HistoricalBattle]:
    """Filter battles. String filters are case-insensitive substring matches, except
    ``war``, which matches whole words (``"world war i"`` excludes World War II).

    ``text`` searches names, campaigns, locations and Wikipedia titles; ``actor``
    and ``commander`` match either side.
    """

    def has(haystack: Iterable[str | None], needle: str) -> bool:
        n = needle.casefold()
        return any(h is not None and n in h.casefold() for h in haystack)

    # Whole-word match so "World War I" does not also match "World War II".
    war_pattern = re.compile(rf"(?<!\w){re.escape(war.strip())}(?!\w)", re.I) if war else None
    out = []
    for b in battles():
        sides = (b.attacker, b.defender)
        if text and not has((b.name, b.campaign, b.location, b.wikipedia_title), text):
            continue
        if war_pattern and not any(war_pattern.search(h) for h in (b.war, b.theater) if h):
            continue
        if actor and not has((x for s in sides for x in (*s.actors, s.force)), actor):
            continue
        if commander and not has((c for s in sides for c in s.commanders), commander):
            continue
        if year_from is not None and b.year < year_from:
            continue
        if year_to is not None and b.year > year_to:
            continue
        if winner is not None and b.winner != winner:
            continue
        if fittable is not None and b.fittable != fittable:
            continue
        out.append(b)
    return out


def to_pandas() -> Any:
    """All battles as a flat ``pandas.DataFrame``, one row per battle (needs ``jomini[pandas]``)."""
    import pandas as pd

    rows = []
    for b in battles():
        row: dict[str, Any] = {}
        for f in fields(b):
            value = getattr(b, f.name)
            if isinstance(value, (Side, Advantages)):
                for sf in fields(value):
                    v = getattr(value, sf.name)
                    row[f"{f.name}_{sf.name}"] = "; ".join(v) if isinstance(v, tuple) else v
            else:
                row[f.name] = value
        row["year"] = b.year
        row["fittable"] = b.fittable
        rows.append(row)
    df = pd.DataFrame(rows).set_index("id")
    for col in df.columns:
        if col.endswith(
            ("_strength", "_casualties", "_cavalry", "_tanks", "_artillery", "_aircraft")
        ):
            df[col] = df[col].astype("Int64")
    return df
