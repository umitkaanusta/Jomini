"""Calibrate the equipment catalogue's kill chances against Ukraine's reported stop rates.

A defence like a large Ukrainian city's (mobile gun teams, Gepards, interceptor drones,
jamming, F-16s, helicopters, IRIS-T SLM, NASAMS and a Patriot battery) faces a typical
night: 125 Shahed-136 over two hours, 10 Kh-101 and 5 Iskander-M. Kill chances against
drones and cruise missiles are scaled until the share stopped matches Ukraine's reports
since September 2024 (87% of drones, 78% of cruise missiles). Paste the printed
multipliers into ``_CALIBRATION`` in ``src/jomini/equipment.py``.

Ballistic missiles are not calibrated. Ukraine stops 26% of them nationally, but most
strike places Patriot doesn't cover: the same city without Patriot stops about that
share with the hand-set kill chances, and with Patriot about three in four. The script
prints both as a check. Medium-range ballistic missiles have no usable calibration data
(interceptor stocks are secret and many Iranian missiles failed on their own).

The force list is illustrative, not Ukraine's real deployment around any city.

    uv run python scripts/calibrate_equipment.py
"""

from __future__ import annotations

import math

from jomini import equipment as eq

REFERENCE = {
    "mobile gun team": 30,
    "Gepard": 4,
    "interceptor drone team": 10,
    "jamming": 2,
    "F-16": 2,
    "Mi-8 / Mi-24": 2,
    "IRIS-T SLM": 1,
    "NASAMS": 1,
    "Patriot PAC-3": 1,
}
TARGET_STOPPED = {"loitering drone": 0.87, "cruise missile": 0.78}
BALLISTIC_NATIONAL = 0.26
TIED = {"loitering drone": ("decoy drone",)}


def night(drones: bool = True, missiles: bool = True) -> list:
    """The reference night. Without drones, the missiles keep their spacing but start
    at once, which makes the simulation much shorter."""
    shift = 0.0 if drones else 1.0
    waves = []
    if drones:
        waves.append(eq.wave("Shahed-136", 125, launch_window=2.0))
    if missiles:
        waves.append(eq.wave("Kh-101", 10, launch_at=1.0 - shift, launch_window=0.5))
        waves.append(eq.wave("Iskander-M", 5, launch_at=2.5 - shift, launch_window=0.1))
    return waves


def stopped(
    multipliers: dict[str, float], drones: bool = True, missiles: bool = True
) -> dict[str, float]:
    eq._CALIBRATION.clear()
    for kind, m in multipliers.items():
        eq._CALIBRATION[kind] = m
        for other in TIED.get(kind, ()):
            eq._CALIBRATION[other] = m
    waves = night(drones, missiles)
    result = eq.defense(REFERENCE).strike(waves)
    kinds = {w.name: w.kind for w in waves}
    return {kinds[w]: 1 - result.leak_fraction(w) for w in result.waves}


def solve(m: dict[str, float], kind: str, target: float, **which: bool) -> None:
    """Bisect the multiplier for one class (stop share rises with it)."""
    lo, hi = math.log(0.02), math.log(20.0)
    for _ in range(18):
        mid = (lo + hi) / 2
        m[kind] = math.exp(mid)
        if stopped(m, **which)[kind] < target:
            lo = mid
        else:
            hi = mid
    m[kind] = math.exp((lo + hi) / 2)


def main() -> None:
    m = dict.fromkeys(TARGET_STOPPED, 1.0)
    # Drones and missiles don't share defences under the default fire rules, so each
    # is calibrated on its own; the missiles together, since Patriot fires at both.
    solve(m, "loitering drone", TARGET_STOPPED["loitering drone"], missiles=False)
    solve(m, "cruise missile", TARGET_STOPPED["cruise missile"], drones=False)
    final = stopped(m)
    for kind, target in TARGET_STOPPED.items():
        print(f"{kind:18s} x{m[kind]:.3f}  stopped {final[kind]:.3f} (target {target})")
    no_patriot = {k: v for k, v in REFERENCE.items() if k != "Patriot PAC-3"}
    for label, force in (("with Patriot", REFERENCE), ("without Patriot", no_patriot)):
        r = eq.defense(force).strike(night(drones=False))
        print(f"ballistic stopped {label}: {1 - r.leak_fraction('Iskander-M'):.3f}"
              f" (Ukraine nationally {BALLISTIC_NATIONAL})")  # fmt: skip
    print("_CALIBRATION = {")
    for kind, v in m.items():
        print(f'    "{kind}": {v:.3f},')
        for other in TIED.get(kind, ()):
            print(f'    "{other}": {v:.3f},')
    print("}")


if __name__ == "__main__":
    main()
