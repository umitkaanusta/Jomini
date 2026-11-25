"""Fit the three-layer air-defence model to Ukraine's nightly Shahed reports.

Nights with 5+ drones launched, Sep 2024 onward, are grouped by raid size relative to
that month's median raid. The model's leak share at each group's size is matched to the
observed share (weighted by its standard error). Free parameters: a common scale on
every layer's pk and the surge exponent; capacity is unlimited (a finite capacity fit
ran to the top of its range). Prints the values used by ``ukraine_defense()``.

    uv run python scripts/fit_ukraine_defense.py
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from datetime import date

import numpy as np
from scipy.optimize import minimize

from jomini import data
from jomini.airdefense import AirDefense, generic_defense

EDGES = (0, 0.35, 0.6, 0.85, 1.15, 1.6, 2.5, math.inf)


def groups() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    raids = data.ukraine_air_raids("drone").select(start=date(2024, 9, 1), min_launched=5)
    launched, stopped = raids.launched.astype(float), raids.stopped.astype(float)
    months = np.array(raids.months)
    median = np.array([np.median(launched[months == m]) for m in months])
    rel = launched / median
    sizes, leak, se = [], [], []
    for lo, hi in itertools.pairwise(EDGES):
        m = (rel >= lo) & (rel < hi)
        w = launched[m] / launched[m].sum()
        share = 1 - stopped[m].sum() / launched[m].sum()
        night = 1 - stopped[m] / launched[m]
        sizes.append(float(np.average(rel[m], weights=launched[m])))
        leak.append(share)
        se.append(math.sqrt(np.sum(w**2 * (night - share) ** 2) * m.sum() / (m.sum() - 1)))
    return np.array(sizes), np.array(leak), np.array(se), float(np.median(launched))


def defense(pk_scale: float, surge: float) -> AirDefense:
    base = generic_defense()
    layers = tuple(
        replace(lay, pk=min(0.99, lay.pk * pk_scale), fire_rate=math.inf, magazine=math.inf)
        for lay in base.layers
    )
    return AirDefense(layers).with_surge(surge)


def main() -> None:
    rel, leak, se, typical = groups()
    sizes = [max(1, round(r * typical)) for r in rel]

    def model(p: np.ndarray) -> np.ndarray:
        d = defense(math.exp(p[0]), p[1])
        return np.array([d.raid(s).leak_fraction for s in sizes])

    def chi2(p: np.ndarray) -> float:
        return float(np.sum(((leak - model(p)) / se) ** 2))

    fit = min(
        (minimize(chi2, x0, method="Nelder-Mead", bounds=[(-1, 0.4), (-0.5, 2)])
         for x0 in ([0.0, 0.2], [0.1, 0.5], [-0.1, 0.0])),
        key=lambda r: r.fun,
    )  # fmt: skip
    pred = model(fit.x)
    print(
        f"typical raid {typical:.0f} drones; pk scale {math.exp(fit.x[0]):.3f}; "
        f"surge {fit.x[1]:.3f}"
    )
    print(
        f"chi2 {fit.fun:.1f} on {len(rel)} groups; "
        f"average miss {100 * np.mean(abs(leak - pred)):.1f} pts"
    )
    for r, o, m in zip(rel, leak, pred, strict=True):
        print(f"  {r:4.2f}x typical  data {100 * o:5.1f}%  model {100 * m:5.1f}%")
    print(
        "pk:",
        [round(min(0.99, lay.pk * math.exp(fit.x[0])), 2) for lay in generic_defense().layers],
    )


if __name__ == "__main__":
    main()
