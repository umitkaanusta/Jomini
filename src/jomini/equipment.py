"""A catalogue of attacking and defending equipment for air-defence simulations.

Pick systems by name and count instead of typing in numbers::

    from jomini import equipment as eq

    defense = eq.defense({"Patriot PAC-3": 1, "IRIS-T SLM": 1, "Gepard": 6, "F-16": 4,
                          "mobile gun team": 30, "interceptor drone team": 10})
    defense.strike([eq.wave("Shahed-136", 120, launch_window=2),
                    eq.wave("Iskander-M", 6, launch_at=2.5)])

**Where the numbers come from.** Speeds, ranges, missiles per battery and costs are
rounded public figures (manufacturer data, defence ministries, think-tank estimates) and
vary by source and variant. Kill chances and firing rates are *not* public. They are
rough defaults per class of system, set by hand. Kill chances against drones and cruise
missiles are then scaled so that a defence like a large Ukrainian city's stops about
the share Ukraine reports since September 2024 (87% of drones, 78% of cruise
missiles; see ``scripts/calibrate_equipment.py``). Treat them as illustrative, not as
the real performance of any system.

Missile batteries fire two interceptors at each target that crosses their zone (one
for upper-tier missile defence, which has time to look and fire again). Medium- and
long-range SAMs and missile defence hold fire on slow drones by default.

**Geometry.** The model is one-dimensional: distance from the target. Each defence is a
band of that distance where it can engage (``deployed``). Area defences (SAMs, lasers,
guns at the target) sit at ``0..range``; forward defences (fighters, mobile gun teams,
interceptor drones, jamming) cover bands along the approach routes. Bands are capped at
300 km, the edge of the simulated airspace.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from jomini.airdefense import AirDefense, Layer, Wave

AttackerKind = Literal[
    "loitering drone",
    "jet drone",
    "decoy drone",
    "combat drone",
    "cruise missile",
    "supersonic missile",
    "ballistic missile",
    "medium-range ballistic missile",
    "hypersonic missile",
    "strike aircraft",
    "stealth aircraft",
    "glide bomb",
    "rocket",
]
DefenderKind = Literal[
    "gun team",
    "anti-aircraft gun",
    "interceptor drone",
    "jamming",
    "fighter",
    "helicopter",
    "laser",
    "microwave",
    "short-range SAM",
    "medium-range SAM",
    "long-range SAM",
    "upper-tier missile defence",
    "naval air defence",
]

ATTACKER_KINDS: tuple[str, ...] = AttackerKind.__args__  # type: ignore[attr-defined]
DEFENDER_KINDS: tuple[str, ...] = DefenderKind.__args__  # type: ignore[attr-defined]
EDGE_KM = 300.0


@dataclass(frozen=True)
class Attacker:
    name: str
    kind: AttackerKind
    origin: str
    speed_kmh: float
    """Typical speed over the last few hundred km (terminal speed for ballistic)."""
    range_km: float
    unit_cost: float
    """Rough cost per round or airframe, USD."""
    aliases: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Defender:
    name: str
    kind: DefenderKind
    origin: str
    range_km: float
    """Engagement range against aircraft-type targets."""
    shots: float
    """Shots per unit before reloading (inf for guns, jamming, lasers)."""
    cost_per_shot: float
    aliases: tuple[str, ...] = ()
    note: str = ""


# ----------------------------------------------------------------- class defaults

# Per unit: shots per hour at a lone target (engage rate), most shots per hour
# (capacity), and the band of distance from the target it covers by default (None:
# from the target out to the system's range).
_DEFENDER_CLASS: dict[str, tuple[float, float, tuple[float, float] | None]] = {
    "gun team": (0.08, 4.0, (0.0, 150.0)),
    "anti-aircraft gun": (0.6, 30.0, (0.0, 40.0)),
    "interceptor drone": (0.25, 4.0, (10.0, 150.0)),
    "jamming": (0.5, math.inf, (0.0, 60.0)),
    "fighter": (0.25, 4.0, (40.0, 300.0)),
    "helicopter": (0.2, 6.0, (10.0, 120.0)),
    "laser": (40.0, 300.0, None),
    "microwave": (60.0, math.inf, None),
    # Missile batteries fire a set number of interceptors per target (_SALVO), so their
    # engage rate here is unused; capacity is how fast they can launch.
    "short-range SAM": (1.0, 600.0, None),
    "medium-range SAM": (1.0, 400.0, None),
    "long-range SAM": (1.0, 600.0, None),
    "upper-tier missile defence": (1.0, 600.0, None),
    "naval air defence": (1.0, 600.0, None),
}

# Kill chance per shot by defender class (rows) and attacker class (columns). 0 means
# the system can't engage that class. Hand-set, then scaled per attacker class by
# scripts/calibrate_equipment.py (see the module docstring).
_PK: dict[str, dict[str, float]] = {
    "gun team": {"loitering drone": 0.35, "decoy drone": 0.35, "combat drone": 0.3,
                 "jet drone": 0.1, "cruise missile": 0.05},
    "anti-aircraft gun": {"loitering drone": 0.55, "decoy drone": 0.55, "combat drone": 0.5,
                          "jet drone": 0.35, "cruise missile": 0.3, "rocket": 0.3},
    "interceptor drone": {"loitering drone": 0.6, "decoy drone": 0.6, "combat drone": 0.5,
                          "jet drone": 0.15},
    "jamming": {"loitering drone": 0.3, "decoy drone": 0.3, "combat drone": 0.2,
                "jet drone": 0.2},
    "fighter": {"loitering drone": 0.6, "decoy drone": 0.6, "combat drone": 0.7,
                "jet drone": 0.6, "cruise missile": 0.6, "supersonic missile": 0.1,
                "strike aircraft": 0.4, "stealth aircraft": 0.1, "glide bomb": 0.05},
    "helicopter": {"loitering drone": 0.5, "decoy drone": 0.5, "combat drone": 0.4,
                   "jet drone": 0.1},
    "laser": {"loitering drone": 0.8, "decoy drone": 0.8, "combat drone": 0.6,
              "jet drone": 0.5, "rocket": 0.6},
    "microwave": {"loitering drone": 0.5, "decoy drone": 0.5},
    "short-range SAM": {"loitering drone": 0.85, "decoy drone": 0.85, "combat drone": 0.85,
                        "jet drone": 0.8, "cruise missile": 0.75, "supersonic missile": 0.3,
                        "strike aircraft": 0.6, "stealth aircraft": 0.2, "glide bomb": 0.3,
                        "rocket": 0.85},
    "medium-range SAM": {"loitering drone": 0.85, "decoy drone": 0.85, "combat drone": 0.85,
                         "jet drone": 0.85, "cruise missile": 0.8, "supersonic missile": 0.4,
                         "ballistic missile": 0.1, "strike aircraft": 0.7,
                         "stealth aircraft": 0.25, "glide bomb": 0.4, "rocket": 0.5},
    "long-range SAM": {"loitering drone": 0.8, "decoy drone": 0.8, "combat drone": 0.85,
                       "jet drone": 0.85, "cruise missile": 0.85, "supersonic missile": 0.55,
                       "ballistic missile": 0.6, "medium-range ballistic missile": 0.4,
                       "hypersonic missile": 0.4, "strike aircraft": 0.8,
                       "stealth aircraft": 0.3, "glide bomb": 0.5, "rocket": 0.4},
    "upper-tier missile defence": {"ballistic missile": 0.7,
                                   "medium-range ballistic missile": 0.75,
                                   "hypersonic missile": 0.3},
    "naval air defence": {"loitering drone": 0.8, "decoy drone": 0.8, "combat drone": 0.85,
                          "jet drone": 0.85, "cruise missile": 0.85, "supersonic missile": 0.6,
                          "ballistic missile": 0.6, "medium-range ballistic missile": 0.6,
                          "hypersonic missile": 0.3, "strike aircraft": 0.8,
                          "stealth aircraft": 0.3, "glide bomb": 0.4},
}  # fmt: skip

# Interceptors fired at each target by missile batteries ("shoot two" for missiles).
_SALVO: dict[str, float] = {
    "short-range SAM": 2.0,
    "medium-range SAM": 2.0,
    "long-range SAM": 2.0,
    "upper-tier missile defence": 1.0,  # shoot, look, shoot again: time allows it
    "naval air defence": 2.0,
}

# What each class fires at by default. Expensive missiles are kept for missiles and
# aircraft: medium- and long-range SAMs and missile defence leave slow drones to guns,
# jamming and interceptor drones. Override with ``layer(..., targets="all")``.
_SLOW_DRONES = ("loitering drone", "decoy drone")
_HOLD_FIRE: dict[str, tuple[str, ...]] = {
    "medium-range SAM": _SLOW_DRONES,
    "long-range SAM": _SLOW_DRONES,
    "upper-tier missile defence": _SLOW_DRONES,
}

# Calibration multipliers per attacker class, from scripts/calibrate_equipment.py.
# Cruise missiles fly low and are seen late, so they are much harder to hit than the
# systems' ranges suggest. Ballistic and other classes keep the hand-set values (see the
# script for why Ukraine's national ballistic figure isn't used).
_CALIBRATION: dict[str, float] = {
    "loitering drone": 1.326,
    "decoy drone": 1.326,
    "cruise missile": 0.320,
}


def kill_chance(defender_kind: str, attacker_kind: str) -> float:
    """Default kill chance per shot of a class of defence against a class of attacker."""
    base = _PK.get(defender_kind, {}).get(attacker_kind, 0.0)
    return min(0.99, base * _CALIBRATION.get(attacker_kind, 1.0))


# ------------------------------------------------------------------- the catalogue

ATTACKERS: tuple[Attacker, ...] = (
    # Drones
    Attacker("Shahed-136", "loitering drone", "Iran", 185, 2000, 35_000, ("Geran-2",)),
    Attacker("Shahed-131", "loitering drone", "Iran", 180, 900, 20_000, ("Geran-1",)),
    Attacker("Shahed-238", "jet drone", "Iran", 550, 1500, 150_000, ("Geran-3",)),
    Attacker("Gerbera", "decoy drone", "Russia", 160, 600, 10_000, (),
             "Decoy with little or no warhead"),
    Attacker("Lancet-3", "loitering drone", "Russia", 110, 70, 35_000),
    Attacker("Bayraktar TB2", "combat drone", "Turkey", 130, 300, 5_000_000, ("TB2",)),
    Attacker("Bayraktar Akinci", "combat drone", "Turkey", 250, 6000, 15_000_000, ("Akinci",)),
    Attacker("Wing Loong II", "combat drone", "China", 280, 4000, 2_000_000),
    Attacker("CH-4", "combat drone", "China", 180, 3500, 4_000_000, ("Rainbow CH-4",)),
    Attacker("CH-5", "combat drone", "China", 220, 6500, 5_000_000, ("Rainbow CH-5",)),
    Attacker("Mohajer-6", "combat drone", "Iran", 180, 2000, 1_000_000),
    Attacker("MQ-9 Reaper", "combat drone", "United States", 310, 1800, 30_000_000,
             ("Reaper", "MQ-9")),
    Attacker("Shahed-107", "loitering drone", "Iran", 160, 1500, 20_000),
    Attacker("Arash-2", "loitering drone", "Iran", 185, 2000, 50_000),
    Attacker("Harop", "loitering drone", "Israel", 185, 1000, 200_000),
    Attacker("Switchblade 600", "loitering drone", "United States", 185, 40, 100_000),
    Attacker("Kargu-2", "loitering drone", "Turkey", 145, 10, 30_000),
    Attacker("Liutyi", "loitering drone", "Ukraine", 150, 2000, 200_000, ("Lyutyi", "AN-196")),
    Attacker("UJ-22 Airborne", "loitering drone", "Ukraine", 160, 800, 100_000, ("UJ-22",)),
    Attacker("Bober", "loitering drone", "Ukraine", 150, 1000, 100_000, ("Beaver",)),
    Attacker("Samad-3", "loitering drone", "Houthis (Yemen)", 200, 1500, 20_000),
    Attacker("Orlan-10", "decoy drone", "Russia", 120, 120, 90_000, (),
             "Reconnaissance drone; counted here as a decoy-like target"),
    Attacker("Bayraktar Kizilelma", "jet drone", "Turkey", 900, 900, 30_000_000,
             ("Kizilelma",)),
    Attacker("TAI Anka", "combat drone", "Turkey", 200, 2500, 10_000_000, ("Anka",)),
    Attacker("Hermes 900", "combat drone", "Israel", 220, 2000, 20_000_000),
    Attacker("Heron TP", "combat drone", "Israel", 300, 7400, 20_000_000, ("Eitan",)),
    Attacker("Orion", "combat drone", "Russia", 200, 1000, 5_000_000, ("Inokhodets",)),
    Attacker("GJ-11", "stealth aircraft", "China", 800, 1500, 0, ("Sharp Sword",),
             "Stealth combat drone"),
    # Cruise missiles
    Attacker("Kh-101", "cruise missile", "Russia", 750, 2800, 13_000_000, ("X-101",)),
    Attacker("Kalibr", "cruise missile", "Russia", 800, 1500, 6_500_000, ("3M14",)),
    Attacker("Iskander-K", "cruise missile", "Russia", 850, 500, 4_000_000, ("9M728",)),
    Attacker("Kh-59", "cruise missile", "Russia", 1000, 290, 1_500_000, ("X-59", "Kh-69")),
    Attacker("Tomahawk", "cruise missile", "United States", 880, 1600, 2_000_000),
    Attacker("Storm Shadow", "cruise missile", "UK / France", 1000, 550, 3_000_000,
             ("SCALP-EG", "SCALP")),
    Attacker("Taurus KEPD 350", "cruise missile", "Germany / Sweden", 1100, 500, 1_500_000,
             ("Taurus",)),
    Attacker("Paveh", "cruise missile", "Iran", 800, 1650, 1_000_000),
    Attacker("CJ-10", "cruise missile", "China", 800, 1500, 2_000_000, ("Changjian-10",)),
    Attacker("Kh-55", "cruise missile", "Soviet Union", 800, 2500, 2_000_000, ("Kh-555",)),
    Attacker("Neptune", "cruise missile", "Ukraine", 900, 300, 1_000_000, ("R-360",)),
    Attacker("Long Neptune", "cruise missile", "Ukraine", 900, 1000, 1_500_000),
    Attacker("Flamingo", "cruise missile", "Ukraine", 900, 3000, 500_000, ("FP-5",)),
    Attacker("JASSM-ER", "cruise missile", "United States", 1000, 925, 1_500_000,
             ("AGM-158B", "JASSM")),
    Attacker("Harpoon", "cruise missile", "United States", 860, 280, 1_500_000),
    Attacker("Exocet", "cruise missile", "France", 1100, 180, 2_000_000),
    Attacker("Hoveyzeh", "cruise missile", "Iran", 800, 1350, 1_000_000),
    Attacker("Quds-1", "cruise missile", "Houthis (Yemen)", 750, 1300, 500_000),
    Attacker("Kh-35", "cruise missile", "Russia", 950, 260, 500_000, ("Uran",)),
    # Supersonic and hypersonic missiles
    Attacker("Kh-22", "supersonic missile", "Russia", 4000, 600, 1_000_000, ("Kh-32", "X-22")),
    Attacker("P-800 Oniks", "supersonic missile", "Russia", 3000, 600, 1_500_000,
             ("Oniks", "Yakhont")),
    Attacker("BrahMos", "supersonic missile", "India / Russia", 3400, 450, 3_000_000),
    Attacker("Kh-31", "supersonic missile", "Russia", 3000, 110, 500_000, ("Kh-31P",)),
    Attacker("YJ-12", "supersonic missile", "China", 3500, 400, 2_000_000),
    Attacker("YJ-18", "supersonic missile", "China", 3000, 540, 2_000_000),
    Attacker("3M22 Zircon", "hypersonic missile", "Russia", 7000, 1000, 5_000_000, ("Zircon",)),
    Attacker("Kinzhal", "hypersonic missile", "Russia", 8000, 2000, 10_000_000,
             ("Kh-47M2", "X-47")),
    Attacker("Fattah-1", "hypersonic missile", "Iran", 9000, 1400, 2_000_000, ("Fattah",)),
    Attacker("DF-17", "hypersonic missile", "China", 9000, 2000, 10_000_000),
    Attacker("Fattah-2", "hypersonic missile", "Iran", 9000, 1500, 3_000_000),
    Attacker("Oreshnik", "hypersonic missile", "Russia", 12000, 5000, 50_000_000, (),
             "Intermediate-range missile with multiple warheads; counts as one target"),
    # Ballistic missiles
    Attacker("Iskander-M", "ballistic missile", "Russia", 6500, 500, 3_000_000, ("9M723",)),
    Attacker("KN-23", "ballistic missile", "North Korea", 6500, 700, 2_000_000, ("Hwasong-11",)),
    Attacker("Fateh-110", "ballistic missile", "Iran", 4500, 300, 500_000, ("Fateh",)),
    Attacker("ATACMS", "ballistic missile", "United States", 3600, 300, 1_500_000),
    Attacker("DF-15", "ballistic missile", "China", 6000, 600, 2_000_000),
    Attacker("Tochka-U", "ballistic missile", "Soviet Union", 4000, 120, 500_000, ("Tochka",)),
    Attacker("S-300 (ground attack)", "ballistic missile", "Russia", 4500, 150, 500_000,
             ("5V55",), "Air-defence missiles fired at ground targets"),
    Attacker("Zolfaghar", "ballistic missile", "Iran", 5500, 700, 1_000_000),
    Attacker("Dezful", "ballistic missile", "Iran", 6000, 1000, 1_500_000),
    Attacker("Burkan-3", "ballistic missile", "Houthis (Yemen)", 6000, 1200, 1_000_000),
    Attacker("PrSM", "ballistic missile", "United States", 5000, 500, 1_500_000),
    Attacker("Hrim-2", "ballistic missile", "Ukraine", 6000, 500, 1_500_000, ("Grom-2",)),
    Attacker("DF-11", "ballistic missile", "China", 5500, 300, 1_000_000),
    Attacker("Emad", "medium-range ballistic missile", "Iran", 10000, 1700, 3_000_000),
    Attacker("Ghadr", "medium-range ballistic missile", "Iran", 10000, 1950, 3_000_000,
             ("Ghadr-110", "Qadr")),
    Attacker("Kheibar Shekan", "medium-range ballistic missile", "Iran", 10000, 1450,
             3_000_000),
    Attacker("Sejjil", "medium-range ballistic missile", "Iran", 11000, 2000, 4_000_000),
    Attacker("DF-21", "medium-range ballistic missile", "China", 11000, 1700, 10_000_000),
    Attacker("Shahab-3", "medium-range ballistic missile", "Iran", 9000, 1300, 2_000_000),
    Attacker("Khorramshahr", "medium-range ballistic missile", "Iran", 11000, 2000,
             5_000_000, ("Khorramshahr-4",)),
    Attacker("Haj Qasem", "medium-range ballistic missile", "Iran", 10000, 1400, 3_000_000),
    Attacker("Palestine-2", "medium-range ballistic missile", "Houthis (Yemen)", 10000,
             2150, 2_000_000),
    Attacker("DF-26", "medium-range ballistic missile", "China", 12000, 4000, 15_000_000),
    # Aircraft on strike missions
    Attacker("F-16 (strike)", "strike aircraft", "United States", 900, 550, 0),
    Attacker("Su-34", "strike aircraft", "Russia", 900, 1100, 0),
    Attacker("F-15E", "strike aircraft", "United States", 950, 1300, 0),
    Attacker("F-35 (strike)", "stealth aircraft", "United States", 900, 1100, 0),
    Attacker("J-20", "stealth aircraft", "China", 1000, 1100, 0),
    Attacker("B-2", "stealth aircraft", "United States", 900, 11000, 0),
    Attacker("Su-25", "strike aircraft", "Soviet Union", 750, 500, 0),
    Attacker("Su-24", "strike aircraft", "Soviet Union", 900, 600, 0),
    Attacker("Rafale (strike)", "strike aircraft", "France", 950, 1800, 0),
    Attacker("Su-57", "stealth aircraft", "Russia", 1000, 1500, 0),
    Attacker("J-35", "stealth aircraft", "China", 1000, 1200, 0),
    Attacker("F-22 (strike)", "stealth aircraft", "United States", 1000, 850, 0),
    # Glide bombs and rockets
    Attacker("KAB with UMPK", "glide bomb", "Russia", 700, 70, 20_000,
             ("UMPK", "FAB-500 UMPK", "KAB"), "Gliding bomb dropped from Su-34s"),
    Attacker("JDAM-ER", "glide bomb", "United States", 700, 70, 50_000),
    Attacker("GBU-39 SDB", "glide bomb", "United States", 700, 110, 40_000, ("SDB",)),
    Attacker("Grad", "rocket", "Soviet Union", 2500, 40, 1_000, ("BM-21",)),
    Attacker("Qassam", "rocket", "Gaza", 1500, 15, 800),
    Attacker("Fajr-5", "rocket", "Iran", 3000, 75, 20_000),
    Attacker("Smerch", "rocket", "Soviet Union", 3500, 90, 50_000, ("BM-30",)),
    Attacker("GMLRS", "rocket", "United States", 3500, 80, 150_000, ("HIMARS",)),
)  # fmt: skip

DEFENDERS: tuple[Defender, ...] = (
    # Guns and teams
    Defender("mobile gun team", "gun team", "generic", 2, math.inf, 300, (),
             "Machine guns or MANPADS on a pickup, spread along drone routes"),
    Defender("Gepard", "anti-aircraft gun", "Germany", 5.5, 30, 3_000, ("Flakpanzer Gepard",),
             "35 mm twin guns; shots are bursts"),
    Defender("Skynex", "anti-aircraft gun", "Germany", 4, 40, 5_000, (),
             "35 mm guns with airburst rounds"),
    Defender("Phalanx", "anti-aircraft gun", "United States", 3.5, 15, 20_000,
             ("C-RAM", "CIWS"), "20 mm close-in gun; shots are bursts"),
    Defender("ZU-23-2", "anti-aircraft gun", "Soviet Union", 2.5, 40, 1_000, ("ZU-23",),
             "Towed twin 23 mm gun, often on a truck"),
    Defender("Shilka", "anti-aircraft gun", "Soviet Union", 2.5, 30, 1_500, ("ZSU-23-4",)),
    Defender("Skyranger 30", "anti-aircraft gun", "Germany", 3, 40, 5_000, ("Skyranger",)),
    Defender("Korkut", "anti-aircraft gun", "Turkey", 4, 40, 4_000),
    Defender("Type 1130", "anti-aircraft gun", "China", 3.5, 20, 20_000, (),
             "30 mm naval close-in gun"),
    Defender("interceptor drone team", "interceptor drone", "Ukraine", 25, 10, 3_000,
             ("Sting", "interceptor drone"), "Cheap fast drones that ram Shaheds"),
    Defender("Coyote", "interceptor drone", "United States", 15, 8, 100_000, (),
             "Jet-powered counter-drone interceptor"),
    Defender("Merops", "interceptor drone", "United States", 20, 10, 15_000),
    Defender("jamming", "jamming", "generic", 30, math.inf, 0, ("electronic warfare", "EW"),
             "Jams navigation so drones crash or lose their way"),
    # Aircraft
    Defender("F-16", "fighter", "United States", 550, 8, 400_000, ("F-16 Fighting Falcon",)),
    Defender("F-15", "fighter", "United States", 1200, 10, 400_000, ("F-15E", "F-15 Eagle")),
    Defender("F-35", "fighter", "United States", 1100, 6, 1_000_000, ("F-35A", "F-35I")),
    Defender("Eurofighter Typhoon", "fighter", "UK / Germany / Italy / Spain", 1400, 8,
             500_000, ("Typhoon",)),
    Defender("Rafale", "fighter", "France", 1800, 8, 500_000),
    Defender("Mirage 2000", "fighter", "France", 1500, 6, 400_000),
    Defender("Su-35", "fighter", "Russia", 1600, 10, 500_000),
    Defender("J-10C", "fighter", "China", 1200, 8, 500_000),
    Defender("J-16", "fighter", "China", 1500, 10, 500_000),
    Defender("MiG-29", "fighter", "Soviet Union", 700, 6, 200_000),
    Defender("Su-27", "fighter", "Soviet Union", 1300, 8, 300_000),
    Defender("Gripen", "fighter", "Sweden", 800, 8, 500_000, ("JAS 39",)),
    Defender("F/A-18", "fighter", "United States", 750, 8, 400_000,
             ("F/A-18E", "Super Hornet")),
    Defender("Yak-52", "helicopter", "Soviet Union", 20, math.inf, 100, (),
             "Old trainer aircraft whose crews shoot drones down with rifles"),
    Defender("Mi-8 / Mi-24", "helicopter", "Soviet Union", 100, math.inf, 1_000,
             ("Mi-8", "Mi-24"), "Door guns against slow drones"),
    Defender("AH-64 Apache", "helicopter", "United States", 400, math.inf, 1_000, ("Apache",)),
    # Lasers
    Defender("Iron Beam", "laser", "Israel", 10, math.inf, 5, (), "100 kW-class laser"),
    Defender("DragonFire", "laser", "UK", 5, math.inf, 15),
    Defender("Tryzub", "laser", "Ukraine", 5, math.inf, 10),
    Defender("Leonidas", "microwave", "United States", 1, math.inf, 1, (),
             "High-power microwave that downs drone swarms at close range"),
    # Short-range SAMs
    Defender("Iron Dome", "short-range SAM", "Israel", 70, 60, 50_000, ("Tamir",),
             "Battery of 3-4 launchers with 20 Tamir each"),
    Defender("IRIS-T SLS", "short-range SAM", "Germany", 12, 8, 400_000),
    Defender("Stinger (Avenger)", "short-range SAM", "United States", 8, 8, 480_000,
             ("Avenger", "Stinger")),
    Defender("Pantsir-S1", "short-range SAM", "Russia", 20, 12, 100_000, ("Pantsir",)),
    Defender("Tor-M2", "short-range SAM", "Russia", 16, 16, 300_000, ("Tor",)),
    Defender("HQ-17", "short-range SAM", "China", 15, 16, 300_000),
    Defender("Strela-10", "short-range SAM", "Soviet Union", 5, 8, 50_000),
    Defender("Osa", "short-range SAM", "Soviet Union", 10, 6, 100_000, ("9K33",)),
    Defender("Tunguska", "short-range SAM", "Soviet Union", 10, 8, 100_000, ("2K22",)),
    Defender("Piorun", "short-range SAM", "Poland", 6, 8, 60_000, (), "MANPADS teams"),
    Defender("Starstreak", "short-range SAM", "UK", 7, 8, 200_000),
    Defender("Crotale", "short-range SAM", "France", 11, 8, 300_000),
    Defender("VL MICA", "short-range SAM", "France", 20, 16, 1_000_000),
    Defender("Hisar-A", "short-range SAM", "Turkey", 15, 16, 400_000),
    Defender("Spyder", "short-range SAM", "Israel", 20, 16, 400_000),
    Defender("SeaRAM", "short-range SAM", "United States", 9, 11, 900_000, ("RAM",)),
    Defender("HQ-7", "short-range SAM", "China", 12, 16, 200_000),
    # Medium-range SAMs
    Defender("NASAMS", "medium-range SAM", "Norway / United States", 40, 18, 1_000_000,
             (), "Three launchers of six AMRAAMs"),
    Defender("IRIS-T SLM", "medium-range SAM", "Germany", 40, 24, 430_000, ("IRIS-T",),
             "Three launchers of eight"),
    Defender("David's Sling", "medium-range SAM", "Israel / United States", 300, 32,
             1_000_000, ("Stunner",), "Also engages short-range ballistic missiles"),
    Defender("Buk-M2", "medium-range SAM", "Russia", 45, 16, 500_000, ("Buk",)),
    Defender("HQ-16", "medium-range SAM", "China", 40, 18, 500_000),
    Defender("Hisar-O", "medium-range SAM", "Turkey", 25, 16, 600_000),
    Defender("Barak MX", "medium-range SAM", "Israel", 150, 24, 1_000_000, ("Barak-8",)),
    Defender("KM-SAM", "medium-range SAM", "South Korea", 40, 24, 1_000_000,
             ("Cheongung",)),
    Defender("Hawk", "medium-range SAM", "United States", 45, 18, 300_000, ("MIM-23",)),
    Defender("Aster 15", "medium-range SAM", "France / Italy", 30, 16, 1_500_000),
    Defender("ESSM", "medium-range SAM", "United States", 50, 32, 1_500_000,
             ("Evolved Sea Sparrow",)),
    Defender("S-350 Vityaz", "medium-range SAM", "Russia", 60, 48, 500_000, ("Vityaz",)),
    Defender("SAMP/T", "long-range SAM", "France / Italy", 120, 48, 2_000_000,
             ("Mamba", "Aster 30"), "Six launchers of eight Aster 30"),
    # Long-range SAMs and lower-tier missile defence
    Defender("Patriot PAC-3", "long-range SAM", "United States", 60, 48, 4_000_000,
             ("Patriot", "PAC-3 MSE"), "Four launchers of twelve PAC-3 MSE"),
    Defender("Patriot PAC-2", "long-range SAM", "United States", 160, 16, 3_000_000,
             ("PAC-2 GEM-T",)),
    Defender("S-400", "long-range SAM", "Russia", 250, 32, 1_000_000, ("Triumf",)),
    Defender("S-300", "long-range SAM", "Soviet Union", 150, 32, 1_000_000),
    Defender("HQ-9", "long-range SAM", "China", 200, 32, 1_000_000),
    Defender("Siper", "long-range SAM", "Turkey", 100, 24, 2_000_000),
    Defender("SAMP/T NG", "long-range SAM", "France / Italy", 150, 48, 2_500_000),
    Defender("HQ-22", "long-range SAM", "China", 170, 24, 800_000),
    # Upper-tier missile defence
    Defender("THAAD", "upper-tier missile defence", "United States", 200, 48, 12_000_000,
             (), "Six launchers of eight; ballistic missiles only"),
    Defender("Arrow 2", "upper-tier missile defence", "Israel / United States", 90, 24,
             3_000_000),
    Defender("Arrow 3", "upper-tier missile defence", "Israel / United States", 2400, 24,
             3_000_000, (), "Intercepts in space"),
    Defender("S-500", "upper-tier missile defence", "Russia", 600, 16, 5_000_000,
             ("Prometey",)),
    # Ships
    Defender("Aegis destroyer (SM-2/SM-6)", "naval air defence", "United States", 240, 40,
             4_000_000, ("Aegis", "SM-6", "SM-2")),
    Defender("Aegis destroyer (SM-3)", "upper-tier missile defence", "United States", 700,
             8, 15_000_000, ("SM-3",), "Ballistic missiles only"),
)  # fmt: skip


def _index(items: Iterable[Attacker | Defender]) -> dict[str, Attacker | Defender]:
    out: dict[str, Attacker | Defender] = {}
    for item in items:
        for key in (item.name, *item.aliases):
            out[key.casefold()] = item
    return out


_ATTACKERS = _index(ATTACKERS)
_DEFENDERS = _index(DEFENDERS)


def attacker(name: str) -> Attacker:
    """Look up an attacker by name or alias (case-insensitive)."""
    try:
        found = _ATTACKERS[name.casefold()]
    except KeyError:
        raise KeyError(f"unknown attacker {name!r}; see equipment.ATTACKERS") from None
    assert isinstance(found, Attacker)
    return found


def defender(name: str) -> Defender:
    """Look up a defence by name or alias (case-insensitive)."""
    try:
        found = _DEFENDERS[name.casefold()]
    except KeyError:
        raise KeyError(f"unknown defence {name!r}; see equipment.DEFENDERS") from None
    assert isinstance(found, Defender)
    return found


def wave(name: str, count: float, *, launch_at: float = 0.0, launch_window: float = 0.0) -> Wave:
    """A wave of ``count`` attackers of one type, arriving at the 300 km edge from
    ``launch_at`` hours over ``launch_window`` hours."""
    a = attacker(name)
    slow = a.speed_kmh < 300
    return Wave(
        a.name,
        count,
        a.speed_kmh,
        launch_at=launch_at,
        launch_window=launch_window,
        diffusion=50.0 if slow else 0.0,
        unit_cost=a.unit_cost,
        kind=a.kind,
    )


def layer(
    name: str,
    count: float = 1,
    *,
    deployed: tuple[float, float] | None = None,
    targets: Iterable[str] | Literal["all"] | None = None,
) -> Layer:
    """``count`` units of a defence as one band of the defence.

    :param deployed: (inner, outer) km from the target it covers. Default: ``0..range``
        for area defences, a class default for forward ones (fighters, gun teams,
        interceptor drones, jamming), capped at 300 km.
    :param targets: Attacker classes it fires at. Default: everything it can hit,
        except that medium- and long-range SAMs and missile defence hold fire on slow
        drones. ``"all"`` lifts that.
    """
    d = defender(name)
    if count <= 0:
        raise ValueError("count must be positive")
    engage, capacity, band = _DEFENDER_CLASS[d.kind]
    inner, outer = deployed or band or (0.0, d.range_km)
    outer = min(outer, EDGE_KM)
    if targets is None:
        allowed = set(ATTACKER_KINDS) - set(_HOLD_FIRE.get(d.kind, ()))
    elif targets == "all":
        allowed = set(ATTACKER_KINDS)
    else:
        allowed = set(targets)
        unknown = allowed - set(ATTACKER_KINDS)
        if unknown:
            raise ValueError(f"unknown attacker classes: {', '.join(sorted(unknown))}")
    pk_against = {
        kind: kill_chance(d.kind, kind) if kind in allowed else 0.0 for kind in ATTACKER_KINDS
    }
    return Layer(
        f"{d.name} x{count:g}" if count != 1 else d.name,
        outer,
        inner,
        pk=pk_against["loitering drone"],
        fire_rate=capacity * count,
        engage_rate=engage * count,
        magazine=d.shots * count,
        cost_per_shot=d.cost_per_shot,
        pk_against=pk_against,
        shots_per_target=_SALVO.get(d.kind),
    )


def defense(units: Mapping[str, float]) -> AirDefense:
    """A defence from ``{name: count}``, e.g. ``{"Patriot PAC-3": 2, "Gepard": 6}``."""
    if not units:
        raise ValueError("need at least one defence")
    return AirDefense(tuple(layer(name, count) for name, count in units.items()))
