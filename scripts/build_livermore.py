"""Build the bundled American Civil War dataset from Livermore (1900).

Source: Thomas L. Livermore, *Numbers and Losses in the Civil War in America,
1861-65* (1900; public domain), as transcribed in jrnold/acw_battle_data
(data licensed ODC-BY 1.0; please cite https://doi.org/10.6084/m9.figshare.1515995).

Usage: uv run python scripts/build_livermore.py [PATH_TO_livermore.csv]
"""

from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/jrnold/acw_battle_data/main/rawdata/livermore1900/livermore.csv"
OUT = Path(__file__).resolve().parent.parent / "src" / "jomini" / "datasets" / "livermore.csv"


def casualties(df: pd.DataFrame, side: str) -> pd.Series:
    """Killed + wounded (or KIA + WIA when the total is missing) + missing/captured."""
    kw = df[f"{side}_kw"].fillna(df[f"{side}_kia"] + df[f"{side}_wia"])
    return (kw + df[f"{side}_miapow"].fillna(0)).astype("Int64")


def main(source: str | None) -> None:
    if source:
        df = pd.read_csv(source)
    else:
        with urllib.request.urlopen(URL) as resp:
            df = pd.read_csv(io.BytesIO(resp.read()))
    winner = df.apply(lambda r: "attacker" if r.result == r.attacker else "defender", axis=1)
    out = pd.DataFrame(
        {
            "id": df.battle_id,
            "name": df.battle_name.str.split(",").str[0].str.strip(),
            "start": df.start_date,
            "end": df.end_date,
            "state": df.state,
            "theater": df.theater,
            "attacker": df.attacker.map({"Union": "USA", "Confederate": "Confederacy"}),
            "winner": winner,
            "us_strength": df.US_str.astype("Int64"),
            "us_casualties": casualties(df, "US"),
            "cs_strength": df.CS_str.astype("Int64"),
            "cs_casualties": casualties(df, "CS"),
            "livermore_page": df.page,
        }
    )
    out.to_csv(OUT, index=False, lineterminator="\n")
    print(f"wrote {len(out)} battles to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
