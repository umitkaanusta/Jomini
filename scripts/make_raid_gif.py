"""Draw docs/raid-v2.gif: a small and a large drone raid against the defence fitted to
Ukraine's nightly reports.

    uv run python scripts/make_raid_gif.py
"""

from __future__ import annotations

from raidviz import Panel, render

from jomini.airdefense import Wave, ukraine_defense

LABELS = {
    "interceptors": "interceptor drones",
    "mobile guns": "mobile gun teams",
    "missiles": "surface-to-air missiles",
}


def raid(size: int) -> list[Wave]:
    d = ukraine_defense()
    return [Wave("drones", size, d.speed, launch_window=1.0, diffusion=d.diffusion)]


def main() -> None:
    render(
        "docs/raid-v2.gif",
        [
            Panel("30 drones", ukraine_defense(), raid(30), LABELS, seed=0),
            Panel("500 drones", ukraine_defense(), raid(500), LABELS, seed=1),
        ],
        slow_step=0.02,
        km_ticks=(0, 40, 150, 300),
    )


if __name__ == "__main__":
    main()
