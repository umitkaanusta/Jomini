"""More battle data: daily WWII attrition, the American Civil War, and the war in Ukraine.

Bundled (public domain):

* :func:`kursk`: day-by-day manpower, tanks, APCs and artillery for both sides
  of the Battle of Kursk, 5-18 July 1943 (Lucas & Turkes 2004, from The Dupuy
  Institute's Kursk Data Base). The rare battle with daily data for both sides,
  so it suits time-series methods (:mod:`jomini.discovery`, :mod:`jomini.dynamics`).
* :func:`livermore`: strengths and losses for 63 major American Civil War battles
  (Livermore 1900), as :class:`~jomini.cdb90.HistoricalBattle` objects that work
  with :mod:`jomini.research` like CDB90 battles.

Fetched at runtime and cached (not bundled):

* :func:`ukraine_general_staff`: Ukraine's General Staff daily claims of Russian
  losses since 25 Feb 2022 (PetroIvaniuk/2022-Ukraine-Russia-War-Dataset, MIT).
  **One side's claims about the other side**, so treat them as an upper-bound series.
* :func:`ukraine_oryx`: visually confirmed equipment losses for both sides,
  derived from Oryx (scarnecchia/oryx_data, CC BY 4.0), updated until August 2025
  (Ukrainian losses only from March 2025).
* :func:`ukraine_oryx_events`: the same kind of data from individual Oryx entries,
  both sides daily, February 2022 to August 2023 (leedrake5/Russia-Ukraine, MIT).
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from jomini.cdb90 import ADVANTAGE_FACTORS, Advantages, HistoricalBattle, Side

# --------------------------------------------------------------------------- Kursk

KurskSide = Literal["german", "soviet"]
KURSK_MEASURES = (
    "manpower", "casualties", "tanks", "tanks_lost", "apcs", "apcs_lost", "artillery",
    "artillery_lost",
)  # fmt: skip


@dataclass(frozen=True)
class DailyBattle:
    """Daily series for a two-sided battle. ``series[side][measure]`` is an array per day."""

    name: str
    dates: tuple[date, ...]
    attacker: str
    defender: str
    series: dict[str, dict[str, NDArray[np.float64]]]
    source: str

    @property
    def days(self) -> NDArray[np.float64]:
        """Days since the first observation (0, 1, 2, ...)."""
        out: NDArray[np.float64] = np.array(
            [(d - self.dates[0]).days for d in self.dates], dtype=float
        )
        return out

    def combat_power(
        self, side: str, weights: Sequence[float] = (1, 5, 20, 40)
    ) -> NDArray[np.float64]:
        """Weighted sum of manpower, APCs, tanks and artillery (the weights used by
        Bracken, Fricker and Lucas & Turkes)."""
        s = self.series[side]
        w_man, w_apc, w_tank, w_art = weights
        out: NDArray[np.float64] = (
            w_man * s["manpower"] + w_apc * s["apcs"] + w_tank * s["tanks"] + w_art * s["artillery"]
        )
        return out

    def trajectory(
        self, measure: str = "combat_power"
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """``(t, attacker, defender)`` for :mod:`jomini.discovery` and friends."""
        if measure == "combat_power":
            red, blue = self.combat_power(self.attacker), self.combat_power(self.defender)
        else:
            if measure not in KURSK_MEASURES:
                raise ValueError(f"unknown measure {measure!r}")
            red, blue = self.series[self.attacker][measure], self.series[self.defender][measure]
        return self.days, red, blue

    def to_pandas(self) -> Any:
        """Long-format DataFrame: one row per day and side (needs ``jomini[pandas]``)."""
        import pandas as pd

        rows = [
            {"date": d, "side": side, **{m: v[i] for m, v in ser.items()}}
            for side, ser in self.series.items()
            for i, d in enumerate(self.dates)
        ]
        return pd.DataFrame(rows)


@cache
def kursk() -> DailyBattle:
    """The southern front of the Battle of Kursk, 5-18 July 1943 (14 days).

    The Germans attack; on 12 July (Prokhorovka) the Soviets lost 414 tanks.
    Manpower is combat manpower; casualties include killed, wounded, missing and
    non-battle; equipment "lost" includes damaged items, some later repaired.
    Daily on-hand counts include reinforcements, so they are not simply the
    previous day minus losses.
    """
    text = resources.files("jomini.datasets").joinpath("kursk.csv").read_text()
    rows = list(csv.DictReader(io.StringIO(text)))
    dates = tuple(sorted({date.fromisoformat(r["date"]) for r in rows}))
    series: dict[str, dict[str, NDArray[np.float64]]] = {}
    for side in ("german", "soviet"):
        mine = sorted((r for r in rows if r["side"] == side), key=lambda r: r["date"])
        series[side] = {m: np.array([float(r[m]) for r in mine]) for m in KURSK_MEASURES}
    return DailyBattle(
        name="Kursk (southern front)",
        dates=dates,
        attacker="german",
        defender="soviet",
        series=series,
        source="Lucas & Turkes (2004), Naval Research Logistics 51, Tables 1-3; "
        "Kursk Data Base, The Dupuy Institute. U.S. Government work.",
    )


# --------------------------------------------------------------------------- Civil War

_NO_ADVANTAGES = Advantages(**dict.fromkeys(ADVANTAGE_FACTORS))
_ACTOR = {"USA": "USA", "Confederacy": "Confederacy"}


def _int(value: str) -> int | None:
    return int(value) if value else None


@cache
def livermore() -> tuple[HistoricalBattle, ...]:
    """63 major American Civil War battles from Livermore (1900).

    Returned as :class:`~jomini.cdb90.HistoricalBattle` with ids 1-63 (their own
    numbering, not CDB90's). ``duration_days`` counts calendar days, not CDB90's
    days of active combat, so per-day coefficients aren't comparable across the two
    datasets; law exponents and effectiveness ratios are.
    """
    text = resources.files("jomini.datasets").joinpath("livermore.csv").read_text()
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        start, end = date.fromisoformat(r["start"]), date.fromisoformat(r["end"])
        us = Side(
            "Union armies",
            ("USA",),
            (),
            _int(r["us_strength"]),
            _int(r["us_casualties"]),
            None,
            None,
            None,
            None,
        )
        cs = Side(
            "Confederate armies",
            ("Confederacy",),
            (),
            _int(r["cs_strength"]),
            _int(r["cs_casualties"]),
            None,
            None,
            None,
            None,
        )
        attacker, defender = (us, cs) if r["attacker"] == "USA" else (cs, us)
        out.append(
            HistoricalBattle(
                id=int(r["id"]),
                name=r["name"],
                war="U.S. Civil War of 1861-1865",
                theater=None,
                campaign=None,
                location=r["state"] or None,
                start=start.isoformat(),
                end=end.isoformat(),
                duration_days=float((end - start).days + 1),
                latitude=None,
                longitude=None,
                front_width_km=None,
                winner=r["winner"],  # type: ignore[arg-type]
                attacker=attacker,
                defender=defender,
                wikipedia_title=None,
                wikidata_id=None,
                defender_posture=None,
                advantages=_NO_ADVANTAGES,
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- Ukraine

_GS_BASE = (
    "https://raw.githubusercontent.com/PetroIvaniuk/2022-Ukraine-Russia-War-Dataset/main/data/"
)
_ORYX_URL = "https://raw.githubusercontent.com/scarnecchia/oryx_data/main/daily_count.csv"
_ORYX_EVENTS_URL = "https://raw.githubusercontent.com/leedrake5/Russia-Ukraine/main/Apps/shinyApp/data/daily_frame.csv"
_UCDP_GED_URL = "https://ucdp.uu.se/downloads/ged/ged261-csv.zip"
Fetcher = Callable[[str], bytes]


def _cache_dir() -> Path:
    root = os.environ.get("JOMINI_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "jomini"
    )
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "jomini (python)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data: bytes = resp.read()
    return data


def _fetch(url: str, max_age_hours: float, fetch: Fetcher | None) -> bytes:
    """Download ``url`` (or reuse a cached copy younger than ``max_age_hours``)."""
    if fetch is not None:
        return fetch(url)
    path = _cache_dir() / url.rsplit("/", 1)[-1]
    if path.exists() and time.time() - path.stat().st_mtime < max_age_hours * 3600:
        return path.read_bytes()
    data = _http_get(url)
    path.write_bytes(data)
    return data


@dataclass(frozen=True)
class LossSeries:
    """A cumulative daily loss series per category (e.g. ``tank``, ``personnel``)."""

    dates: tuple[date, ...]
    cumulative: dict[str, NDArray[np.float64]]
    source: str
    starts_at_zero: bool = False
    """True when the cumulative count began at zero on the first date (e.g. counts built
    from individual events), so the first period's losses are the first value itself."""

    @property
    def days(self) -> NDArray[np.float64]:
        out: NDArray[np.float64] = np.array(
            [(d - self.dates[0]).days for d in self.dates], dtype=float
        )
        return out

    def daily(self, category: str) -> NDArray[np.float64]:
        """New losses per observation interval (first difference of the cumulative)."""
        c = self.cumulative[category]
        out: NDArray[np.float64] = np.diff(c, prepend=0.0 if self.starts_at_zero else c[0])
        return out

    def resample(
        self, every_days: int, start: date | None = None, end: date | None = None
    ) -> LossSeries:
        """Cumulative totals on a regular grid (carrying the last known total forward).

        Use it to put two sources, or two sides recorded on different dates, on
        common periods before comparing them.
        """
        if every_days < 1:
            raise ValueError("every_days must be at least 1")
        first = start or self.dates[0]
        last = end or self.dates[-1]
        known = np.array([(d - first).days for d in self.dates])
        grid = np.arange(0, (last - first).days + 1, every_days)
        idx = np.searchsorted(known, grid, side="right") - 1
        dates = tuple(date.fromordinal(first.toordinal() + int(g)) for g in grid)
        cumulative = {
            k: np.where(idx >= 0, v[np.clip(idx, 0, None)], 0.0) for k, v in self.cumulative.items()
        }
        return LossSeries(dates, cumulative, self.source, self.starts_at_zero)

    def to_pandas(self) -> Any:
        import pandas as pd

        return pd.DataFrame(self.cumulative, index=pd.DatetimeIndex(self.dates, name="date"))


def ukraine_general_staff(*, max_age_hours: float = 24, fetch: Fetcher | None = None) -> LossSeries:
    """Russian losses claimed daily by Ukraine's General Staff, from 25 Feb 2022.

    Categories include ``personnel``, ``tank``, ``APC``, ``field artillery``,
    ``drone`` and more. These are one side's claims, unverified and generally
    higher than visually confirmed counts (see :func:`ukraine_oryx`). Downloaded
    from PetroIvaniuk/2022-Ukraine-Russia-War-Dataset (MIT) and cached for
    ``max_age_hours`` under ``~/.cache/jomini`` (or ``$JOMINI_CACHE``).
    """
    personnel = json.loads(_fetch(_GS_BASE + "russia_losses_personnel.json", max_age_hours, fetch))
    equipment = json.loads(_fetch(_GS_BASE + "russia_losses_equipment.json", max_age_hours, fetch))
    by_date: dict[str, dict[str, float]] = {}
    for row in equipment:
        by_date.setdefault(row["date"], {}).update(
            {k: float(v) for k, v in row.items() if isinstance(v, (int, float)) and k != "day"}
        )
    for row in personnel:
        by_date.setdefault(row["date"], {}).update(
            {k: float(v) for k, v in row.items() if isinstance(v, (int, float)) and k != "day"}
        )
    dates = tuple(date.fromisoformat(d) for d in sorted(by_date))
    categories = sorted({k for v in by_date.values() for k in v})
    cumulative = {
        c: _forward_fill(np.array([by_date[d.isoformat()].get(c, np.nan) for d in dates]))
        for c in categories
    }
    return LossSeries(dates, cumulative, "Ukrainian General Staff via PetroIvaniuk (MIT)")


def ukraine_oryx(
    side: Literal["Russia", "Ukraine"],
    *,
    max_age_hours: float = 24 * 7,
    fetch: Fetcher | None = None,
) -> LossSeries:
    """Visually confirmed equipment losses (destroyed, abandoned, captured, damaged).

    Weekly snapshots per equipment type, from scarnecchia/oryx_data (CC BY 4.0),
    derived from Oryx; no longer updated after August 2025. Category ``"All Types"``
    holds the total.
    """
    if side not in ("Russia", "Ukraine"):
        raise ValueError("side must be 'Russia' or 'Ukraine'")
    text = _fetch(_ORYX_URL, max_age_hours, fetch).decode()
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r["country"] == side]
    dates = tuple(sorted({date.fromisoformat(r["date_recorded"]) for r in rows}))
    index = {d: i for i, d in enumerate(dates)}
    cumulative: dict[str, NDArray[np.float64]] = {}
    for r in rows:
        series = cumulative.setdefault(r["equipment_type"], np.full(len(dates), np.nan))
        series[index[date.fromisoformat(r["date_recorded"])]] = float(r["type_total"])
    cumulative = {k: _forward_fill(v) for k, v in cumulative.items()}
    if "All Types" not in cumulative:
        cumulative["All Types"] = np.sum(list(cumulative.values()), axis=0)
    return LossSeries(dates, cumulative, "Oryx via scarnecchia/oryx_data (CC BY 4.0)")


def ukraine_oryx_events(
    side: Literal["Russia", "Ukraine"],
    *,
    max_age_hours: float = 24 * 30,
    fetch: Fetcher | None = None,
) -> LossSeries:
    """Visually confirmed equipment losses, daily by event date, Feb 2022 to Aug 2023.

    Built from individual Oryx entries (leedrake5/Russia-Ukraine, MIT), so both
    sides are on the same daily grid, unlike :func:`ukraine_oryx`. Categories are
    Oryx classes (``"Tanks"``, ``"Infantry Fighting Vehicles"``, ...) plus
    ``"All Types"``. Dates are when a loss was documented, which can lag the event.
    """
    if side not in ("Russia", "Ukraine"):
        raise ValueError("side must be 'Russia' or 'Ukraine'")
    text = _fetch(_ORYX_EVENTS_URL, max_age_hours, fetch).decode()
    rows = list(csv.DictReader(io.StringIO(text)))
    parsed = []
    for r in rows:
        try:
            month, day, year = (int(x) for x in r["Date"].split("/"))
        except (ValueError, KeyError):
            continue
        parsed.append((r["country"], date(year, month, day), r.get("class") or "NA"))
    if not parsed:
        raise ValueError("no dated events in the source")
    first, last = min(p[1] for p in parsed), max(p[1] for p in parsed)
    n = (last - first).days + 1
    dates = tuple(date.fromordinal(first.toordinal() + i) for i in range(n))
    counts: dict[str, NDArray[np.float64]] = {"All Types": np.zeros(n)}
    for country, when, cls in parsed:
        if country != side:
            continue
        i = (when - first).days
        counts["All Types"][i] += 1
        if cls != "NA":
            counts.setdefault(cls, np.zeros(n))[i] += 1
    cumulative = {k: np.cumsum(v) for k, v in counts.items()}
    return LossSeries(
        dates, cumulative, "Oryx entries via leedrake5/Russia-Ukraine (MIT)", starts_at_zero=True
    )


# --------------------------------------------------------------------------- UCDP GED

UCDP_GED_VERSION = "26.1"
_GED_STR = ("conflict_name", "dyad_name", "side_a", "side_b", "country")
_GED_NUM = ("deaths_a", "deaths_b", "deaths_civilians", "deaths_unknown", "best", "low",
            "high", "latitude", "longitude", "type_of_violence", "date_prec")  # fmt: skip


@dataclass(frozen=True)
class ConflictEvents:
    """UCDP GED events: one row per event of organised violence.

    ``deaths_a`` / ``deaths_b`` are deaths on ``side_a`` / ``side_b`` (for state-based
    conflicts side_a is a government); ``best``/``low``/``high`` are total-death
    estimates including civilians and unknown. ``type_of_violence``: 1 state-based,
    2 non-state, 3 one-sided violence against civilians.
    """

    start: NDArray[np.int64]
    """Event start as days since 1970-01-01."""
    end: NDArray[np.int64]
    strings: dict[str, NDArray[np.object_]]
    numbers: dict[str, NDArray[np.float64]]

    def __len__(self) -> int:
        return int(self.start.size)

    def __getitem__(self, name: str) -> NDArray[Any]:
        if name in self.strings:
            return self.strings[name]
        return self.numbers[name]

    def select(
        self,
        *,
        country: str | None = None,
        conflict: str | None = None,
        type_of_violence: int | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> ConflictEvents:
        """Filter by country or conflict name (case-insensitive substring), type, dates."""
        keep = np.ones(len(self), bool)
        for field_, needle in (("country", country), ("conflict_name", conflict)):
            if needle:
                n = needle.casefold()
                keep &= np.array([n in str(v).casefold() for v in self.strings[field_]])
        if type_of_violence is not None:
            keep &= self.numbers["type_of_violence"] == type_of_violence
        if start is not None:
            keep &= self.end >= _epoch_day(start)
        if end is not None:
            keep &= self.start <= _epoch_day(end)
        return ConflictEvents(
            self.start[keep],
            self.end[keep],
            {k: v[keep] for k, v in self.strings.items()},
            {k: v[keep] for k, v in self.numbers.items()},
        )

    def daily(
        self, value: str = "events", start: date | None = None, end: date | None = None
    ) -> LossSeries:
        """Daily totals of ``value`` ("events" or a numeric column such as ``best``,
        ``deaths_a``). Events spanning several days are spread evenly over them."""
        if not len(self):
            raise ValueError("no events selected")
        if value != "events" and value not in self.numbers:
            raise ValueError(f"unknown value {value!r}")
        first = _epoch_day(start) if start else int(self.start.min())
        last = _epoch_day(end) if end else int(self.end.max())
        n = last - first + 1
        totals = np.zeros(n)
        weights = np.ones(len(self)) if value == "events" else np.nan_to_num(self.numbers[value])
        for s0, e0, w in zip(self.start, self.end, weights, strict=True):
            lo, hi = max(int(s0), first), min(int(e0), last)
            if hi < lo or w == 0:
                continue
            span = int(e0) - int(s0) + 1
            totals[lo - first : hi - first + 1] += w / span
        dates = tuple(date.fromordinal(_EPOCH + first + i) for i in range(n))
        return LossSeries(
            dates, {value: np.cumsum(totals)}, f"UCDP GED {UCDP_GED_VERSION}", starts_at_zero=True
        )


_EPOCH = date(1970, 1, 1).toordinal()


def _epoch_day(d: date) -> int:
    return d.toordinal() - _EPOCH


@cache
def _ged_cached(path: str, mtime: float) -> ConflictEvents:
    import zipfile

    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
            starts: list[int] = []
            ends: list[int] = []
            strs: dict[str, list[str]] = {k: [] for k in _GED_STR}
            nums: dict[str, list[float]] = {k: [] for k in _GED_NUM}
            for row in reader:
                starts.append(_epoch_day(date.fromisoformat(row["date_start"][:10])))
                ends.append(_epoch_day(date.fromisoformat(row["date_end"][:10])))
                for k in _GED_STR:
                    strs[k].append(row[k])
                for k in _GED_NUM:
                    v = row[k]
                    nums[k].append(float(v) if v not in ("", "NA") else np.nan)
    return ConflictEvents(
        np.array(starts, dtype=np.int64),
        np.array(ends, dtype=np.int64),
        {k: np.array(v, dtype=object) for k, v in strs.items()},
        {k: np.array(v, dtype=float) for k, v in nums.items()},
    )


def ucdp_ged(*, max_age_hours: float = 24 * 90, fetch: Fetcher | None = None) -> ConflictEvents:
    """UCDP Georeferenced Event Dataset 26.1: about 418,000 events of organised violence
    worldwide, 1989-2025 (CC BY 4.0; please cite Sundberg & Melander 2013 and Davies et
    al. 2026). Downloads a ~39 MB zip once and caches it; parsing takes a few seconds,
    then results are cached in memory for the session.
    """
    if fetch is not None:
        tmp = _cache_dir() / "ged-injected.zip"
        tmp.write_bytes(fetch(_UCDP_GED_URL))
        return _ged_cached(str(tmp), tmp.stat().st_mtime)
    path = _cache_dir() / _UCDP_GED_URL.rsplit("/", 1)[-1]
    if not path.exists() or time.time() - path.stat().st_mtime > max_age_hours * 3600:
        path.write_bytes(_http_get(_UCDP_GED_URL))
    return _ged_cached(str(path), path.stat().st_mtime)


def _forward_fill(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = values.copy()
    last = 0.0
    for i, v in enumerate(out):
        if np.isnan(v):
            out[i] = last
        else:
            last = v
    return out
