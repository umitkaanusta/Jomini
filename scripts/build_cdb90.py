"""Build the bundled CDB90 dataset from jrnold/CDB90 (CC0).

Usage:  uv run python scripts/build_cdb90.py [SOURCE_DIR]

Without SOURCE_DIR the flat tables are downloaded from the pinned release tag.
Writes src/jomini/datasets/cdb90.csv.gz and cdb90_narratives.csv.gz.
"""

from __future__ import annotations

import csv
import gzip
import io
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

VERSION = "v10.8.0"
URL = "https://raw.githubusercontent.com/jrnold/CDB90/{version}/data/flat/{table}.csv"
OUT = Path(__file__).resolve().parent.parent / "src" / "jomini" / "datasets"

_ROMAN = re.compile(
    r"\b(?=[MDCLXVI]+\b)M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b", re.I
)
_ABBREVIATIONS = {"US", "CS", "USSR", "UK", "UN", "SS", "ROK", "NVA", "ARVN", "BEF", "AEF", "IDF"}
_SMALL = {"of", "the", "and", "at", "on", "in", "de", "la", "le", "du", "von", "der", "to"}


def titlecase(text: str) -> str:
    """'OBOYAN-KURSK (PHASE II)' -> 'Oboyan-Kursk (Phase II)'."""
    words = text.lower().split(" ")
    out = []
    for i, word in enumerate(words):
        if i > 0 and word in _SMALL:
            out.append(word)
            continue
        bare = re.sub(r"[^a-z]", "", word)
        if bare.upper() in _ABBREVIATIONS or (len(bare) > 1 and not re.search("[aeiouy0-9]", word)):
            out.append(word.upper())
            continue
        word = re.sub(r"(^|[-(/\"])([a-z])", lambda m: m.group(1) + m.group(2).upper(), word)
        word = re.sub(r"'S\b", "'s", word)
        out.append(word)
    result = " ".join(out)
    return _ROMAN.sub(lambda m: m.group(0).upper() if m.group(0) else "", result)


def clean_text(text: object) -> str:
    """Strip OCR uncertainty markers (~~word~~) from narratives."""
    if not isinstance(text, str):
        return ""
    return text.replace("~~", "").strip()


def read(source: str | None, table: str) -> pd.DataFrame:
    if source:
        return pd.read_csv(Path(source) / f"{table}.csv")
    with urllib.request.urlopen(URL.format(version=VERSION, table=table)) as resp:
        return pd.read_csv(io.BytesIO(resp.read()))


WINNER = {1: "attacker", -1: "defender", 0: "draw"}
# Analysts' ratings of the attacker's relative advantage (negative = defender's advantage).
ADVANTAGES = {
    "surpa": "surprise",
    "aeroa": "air_superiority",
    "leada": "leadership",
    "trnga": "training",
    "morala": "morale",
    "logsa": "logistics",
    "momnta": "momentum",
    "intela": "intelligence",
    "techa": "technology",
    "inita": "initiative",
}
POSTURE = {"HD": "hasty", "PD": "prepared", "FD": "fortified", "DL": "delay", "WD": "withdrawal"}
SIDE_COLUMNS = {
    "nam": "force",
    "actors": "actors",
    "commander_names": "commanders",
    "str": "strength",
    "cas": "casualties",
    "cav": "cavalry",
    "tank": "tanks",
    "arty": "artillery",
    "fly": "aircraft",
}


def build(source: str | None) -> None:
    battles = read(source, "battles")
    sides = read(source, "sides")
    narratives = read(source, "narratives")

    table = pd.DataFrame(
        {
            "id": battles.isqno,
            "name": battles.name.map(titlecase),
            "wikipedia_title": battles.wikipedia_title,
            "wikidata_id": battles.wikidata_id,
            "war": battles.war4,
            "theater": battles.war4_theater,
            "campaign": battles.campgn.map(titlecase, na_action="ignore"),
            "location": battles.locn.map(titlecase, na_action="ignore"),
            "start": battles.datetime_min,
            "end": battles.datetime_max,
            "duration_days": battles.duration_active_days,
            "latitude": battles.latitude,
            "longitude": battles.longitude,
            "front_width_km": battles.front_width_attacker,
            "winner": battles.wina.map(WINNER),
            "defender_posture": battles.post1.map(POSTURE),
        }
    )
    for src, dst in ADVANTAGES.items():
        table[f"adv_{dst}"] = battles[src].astype("Int64")
    for role, is_attacker in (("attacker", True), ("defender", False)):
        side = sides[sides.is_attacker == is_attacker].set_index("isqno")
        for src, dst in SIDE_COLUMNS.items():
            values = side[src].reindex(table.id).to_numpy()
            table[f"{role}_{dst}"] = values
        table[f"{role}_force"] = table[f"{role}_force"].map(titlecase, na_action="ignore")
        table[f"{role}_commanders"] = table[f"{role}_commanders"].map(titlecase, na_action="ignore")

    for col in [c for c in table.columns if c.endswith(("_strength", "_casualties"))] + [
        c for c in table.columns if c.endswith(("_cavalry", "_tanks", "_artillery", "_aircraft"))
    ]:
        table[col] = table[col].astype("Int64")

    text = pd.DataFrame(
        {
            "id": narratives.isqno,
            "title": narratives.hero_title,
            "narrative": narratives.narrative.map(clean_text),
            "significance": narratives.significance.map(clean_text),
        }
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for frame, name in ((table, "cdb90.csv.gz"), (text, "cdb90_narratives.csv.gz")):
        buf = io.StringIO()
        frame.to_csv(buf, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        # mtime=0 keeps the output byte-for-byte reproducible.
        with open(OUT / name, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
            gz.write(buf.getvalue().encode())
    print(f"wrote {len(table)} battles, {len(text)} narratives to {OUT} (CDB90 {VERSION})")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None)
