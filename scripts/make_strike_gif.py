"""Draw docs/strike.gif: 120 medium-range ballistic missiles against an Israel-style
layered defence, alone and with drones and cruise missiles sent ahead of them.

The defence is built from the equipment catalogue and is illustrative, not Israel's real
deployment.

    uv run python scripts/make_strike_gif.py
"""

from __future__ import annotations

from raidviz import Panel, render

from jomini import equipment as eq

DEFENCE = {
    "Arrow 3": 3,
    "Arrow 2": 3,
    "Aegis destroyer (SM-3)": 2,
    "David's Sling": 1,
    "Iron Dome": 6,
    "F-35": 4,
    "F-15": 6,
}
SHORT = {"Emad": "missiles", "Shahed-136": "drones", "Paveh": "cruise"}


def missiles() -> list:
    return [eq.wave("Emad", 120, launch_at=2.0, launch_window=0.2)]


def drones_first() -> list:
    return [
        eq.wave("Shahed-136", 170, launch_window=1.0),
        eq.wave("Paveh", 30, launch_at=1.4, launch_window=0.3),
        *missiles(),
    ]


def main() -> None:
    labels = {eq.layer(n, c).name: eq.layer(n, c).name for n, c in DEFENCE.items()}
    labels["Aegis destroyer (SM-3) x2"] = "Aegis SM-3 x2"
    render(
        "docs/strike.gif",
        [
            Panel("120 ballistic missiles", eq.defense(DEFENCE), missiles(), labels, SHORT, 0),
            Panel(
                "The same, behind drones and cruise missiles",
                eq.defense(DEFENCE),
                drones_first(),
                labels,
                SHORT,
                1,
            ),
        ],
        busy=lambda t: 1.95 < t < 2.25,
        slow_step=0.025,
        km_ticks=(0, 40, 70, 90, 150, 300),
    )


if __name__ == "__main__":
    main()
