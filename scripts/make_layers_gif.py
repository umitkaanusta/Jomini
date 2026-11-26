"""Draw docs/cheap-layers-v2.gif: the same strike against missile batteries alone, and
against the same batteries behind cheap guns and interceptor drones.

    uv run python scripts/make_layers_gif.py
"""

from __future__ import annotations

from raidviz import Panel, render

from jomini import equipment as eq
from jomini.airdefense import AirDefense

CHEAP = {"Gepard": 6, "mobile gun team": 30, "interceptor drone team": 10}
SHORT = {"Shahed-136": "drones", "Iskander-M": "Iskanders"}
LABELS = {
    "Patriot PAC-3": "Patriot",
    "mobile gun team x30": "gun teams x30",
    "interceptor drone team x10": "interceptor drones x10",
}


def strike() -> list:
    return [
        eq.wave("Shahed-136", 100, launch_window=1.5),
        eq.wave("Iskander-M", 10, launch_at=2.5, launch_window=0.05),
    ]


def batteries() -> list:
    # Without cheaper layers, the missile batteries have to fire at the drones too.
    return [eq.layer("NASAMS", targets="all"), eq.layer("Patriot PAC-3", targets="all")]


def main() -> None:
    cheap = AirDefense((*batteries(), *(eq.layer(n, c) for n, c in CHEAP.items())), edge=300)
    render(
        "docs/cheap-layers-v2.gif",
        [
            Panel(
                "Missile batteries only",
                AirDefense(tuple(batteries()), edge=300),
                strike(),
                LABELS,
                SHORT,
                0,
            ),
            Panel("The same, behind cheap guns and drones", cheap, strike(), LABELS, SHORT, 1),
        ],
        busy=lambda t: 2.45 < t < 2.6,
        km_ticks=(0, 40, 60, 150, 300),
    )


if __name__ == "__main__":
    main()
