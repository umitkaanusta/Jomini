"""Build the bundled Battle of Kursk daily dataset.

Source: Lucas, T. W. and Turkes, T. (2004), "Fitting Lanchester Equations to the
Battles of Kursk and Ardennes", Naval Research Logistics 51(1), pp. 101-102,
Tables 1-3 (from The Dupuy Institute's Kursk Data Base). A work of the U.S.
Government (17 U.S.C. 101), not subject to copyright in the United States:
https://hdl.handle.net/10945/44169

Usage: uv run python scripts/build_kursk.py PATH_TO_EXTRACTED_TEXT
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "jomini" / "datasets" / "kursk.csv"
ROW = re.compile(r"^\s*(\d{1,2}) July\s+([\d\s]+)$")


def table(text: str, title: str, ncols: int) -> list[tuple[int, list[int]]]:
    start = text.index(title)
    rows: list[tuple[int, list[int]]] = []
    for line in text[start:].splitlines()[1:]:
        m = ROW.match(line)
        if m:
            values = [int(v) for v in m.group(2).split()]
            if len(values) != ncols:
                raise ValueError(f"{title}: expected {ncols} values in {line!r}")
            rows.append((int(m.group(1)), values))
        elif rows and len(rows) == 14:
            break
    if [d for d, _ in rows] != list(range(5, 19)):
        raise ValueError(f"{title}: expected days 5-18 July, got {[d for d, _ in rows]}")
    return rows


def main(path: str) -> None:
    text = Path(path).read_text()
    manpower = table(text, "Table 1. Combat manpower for both sides.", 4)
    german = table(text, "Table 2. Daily German on hand and loss data", 6)
    soviet = table(text, "Table 3. Daily Soviet on hand and loss data", 6)
    header = [
        "date",
        "side",
        "manpower",
        "casualties",
        "tanks",
        "tanks_lost",
        "apcs",
        "apcs_lost",
        "artillery",
        "artillery_lost",
    ]
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for (day, mp), (_, ge), (_, so) in zip(manpower, german, soviet, strict=True):
            date = f"1943-07-{day:02d}"
            w.writerow([date, "soviet", mp[0], mp[1], *so])
            w.writerow([date, "german", mp[2], mp[3], *ge])
    print(f"wrote 28 rows to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
