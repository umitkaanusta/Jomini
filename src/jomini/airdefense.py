"""Drone and missile raids against layered air defences.

A spatial Lanchester model of a raid flying through defence layers, with saturation,
finite magazines and cost exchange, plus a statistical test for saturation in real
interception data (see :func:`jomini.data.ukraine_air_raids`).

**Model** (after Randow et al. 2026, reduced to one dimension). Attackers are a
density ``u(x, t)`` per km at distance ``x`` from the target, flying inbound at
speed ``v`` and spreading out with diffusion ``D`` (stragglers, routing)::

    du/dt = v du/dx + D d2u/dx2 - sum_k pk_k * s_k(t) * 1[x in layer k] * u / N_k(t)

``N_k`` is the number of attackers inside layer ``k``. Its shots per hour follow a
Holling type II curve, ``s_k = F e N / (F + e N)``: with few attackers each one is
engaged at rate ``e`` (kills proportional to attackers, as in Lanchester's linear
law); with many, the layer fires at its capacity ``F`` and kills stop growing
(Lanchester's aimed-fire regime, where the attacker's numbers don't help the
defender). Each layer draws on a finite magazine. Attackers reaching ``x = 0`` leak.

**Proposition (no economies of scale in defence).** Suppose each attacker's kill
hazard is a non-increasing function of how many attackers are present (capacity
limits, finite magazines, distraction) and doesn't otherwise depend on raid size.
Then the expected leak *fraction* is non-decreasing in raid size: the leaker count
grows at least proportionally, elasticity >= 1.

*Proof sketch (well-mixed layer).* With ``dN/dt = -h(N, t) N``, solutions from larger
starting sizes stay larger (uniqueness: trajectories can't cross). So the hazard
along the larger raid's path is pointwise lower, and each exchangeable attacker's
survival probability ``exp(-integral h)`` is higher. Magazines strengthen this: a
larger raid draws more shots earlier, so later hazards are lower still. The spatial
model satisfies the same comparison layer by layer, and :func:`leakage_curve`
verifies it numerically.

So a defence whose interception rate *rises* with raid size contradicts every
fixed-capacity model. That needs a mechanism outside this class: forces surged on
warning of a big raid (``Layer.surge``), a decoy share that grows with raid size,
routing, or reporting. :func:`saturation_test` measures the relationship in data.
Ukraine's Air Force nightly reports show interception odds *rising* with raid size
within the same month (see ``docs`` and the MCP tool ``air_defense_saturation_test``).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit

# --------------------------------------------------------------------------- model


@dataclass(frozen=True)
class Layer:
    """One band of air defence, between ``inner`` and ``outer`` km from the target.

    :param pk: Kill probability per shot.
    :param fire_rate: Maximum shots per hour (crews, launchers, guidance channels);
        ``inf`` for a layer that never saturates.
    :param engage_rate: Shots per hour at each attacker when the layer isn't saturated.
    :param magazine: Shots available (``inf`` for guns and EW).
    :param cost_per_shot: For cost exchange.
    :param surge: Elasticity of the layer's capacity with respect to raid size:
        ``fire_rate`` and ``engage_rate`` are multiplied by
        ``(raid / surge_reference) ** surge``. 0 means a fixed defence; positive
        values model forces moved in on warning of bigger raids.
    """

    name: str
    outer: float
    inner: float
    pk: float
    fire_rate: float
    engage_rate: float
    magazine: float = math.inf
    cost_per_shot: float = 0.0
    surge: float = 0.0
    surge_reference: float = 100.0
    pk_against: Mapping[str, float] | None = None
    """For :meth:`AirDefense.strike`: kill chance per shot against each wave by name
    (missing waves use ``pk``; 0 means the layer can't engage that wave)."""
    shots_per_target: float | None = None
    """If set, the layer fires this many shots at each target that crosses it (missile
    batteries' "shoot two" doctrine) instead of firing at ``engage_rate`` for as long
    as the target is in range. Its engage rate then scales with the target's speed."""

    def __post_init__(self) -> None:
        if not 0 <= self.inner < self.outer:
            raise ValueError(f"layer {self.name!r}: need 0 <= inner < outer")
        if not 0 <= self.pk <= 1:
            raise ValueError(f"layer {self.name!r}: pk must be in [0, 1]")
        if self.fire_rate <= 0 or self.engage_rate <= 0:
            raise ValueError(f"layer {self.name!r}: rates must be positive")
        if self.magazine < 0 or self.cost_per_shot < 0 or self.surge_reference <= 0:
            raise ValueError(f"layer {self.name!r}: magazine and costs must be non-negative")
        if self.shots_per_target is not None and self.shots_per_target <= 0:
            raise ValueError(f"layer {self.name!r}: shots_per_target must be positive")
        if self.pk_against and not all(0 <= v <= 1 for v in self.pk_against.values()):
            raise ValueError(f"layer {self.name!r}: pk_against values must be in [0, 1]")

    def engage_rate_for(self, speed: float) -> float:
        """Shots per hour at a lone target moving at ``speed`` km/h."""
        if self.shots_per_target is None:
            return self.engage_rate
        return self.shots_per_target * speed / (self.outer - self.inner)

    def pk_for(self, wave: str, kind: str = "") -> float:
        """Kill chance per shot against ``wave`` (looked up by wave name, then by the
        wave's ``kind``, e.g. "ballistic missile")."""
        if self.pk_against is None:
            return self.pk
        if wave in self.pk_against:
            return self.pk_against[wave]
        return self.pk_against.get(kind, self.pk)


@dataclass(frozen=True)
class LayerReport:
    name: str
    shots: float
    kills: float
    magazine_left: float
    cost: float


@dataclass(frozen=True)
class RaidResult:
    launched: float
    leaked: float
    layers: tuple[LayerReport, ...]
    attacker_cost: float
    t: NDArray[np.float64] = field(repr=False)
    """Hours since the first launch."""
    leak_rate: NDArray[np.float64] = field(repr=False)
    """Attackers reaching the target per hour."""

    @property
    def leak_fraction(self) -> float:
        return self.leaked / self.launched if self.launched else 0.0

    @property
    def killed(self) -> float:
        return sum(r.kills for r in self.layers)

    @property
    def defender_cost(self) -> float:
        return sum(r.cost for r in self.layers)

    @property
    def cost_exchange(self) -> float:
        """Defender spending per unit of attacker spending (> 1 favours the attacker)."""
        return self.defender_cost / self.attacker_cost if self.attacker_cost else math.nan

    def __str__(self) -> str:
        parts = ", ".join(f"{r.name} {r.kills:.0f} kills/{r.shots:.0f} shots" for r in self.layers)
        return (
            f"{self.launched:.0f} launched, {self.leaked:.1f} leaked "
            f"({100 * self.leak_fraction:.1f}%); {parts}"
        )


@dataclass(frozen=True)
class AirDefense:
    """Layered defence of one target against raids flying in at ``speed`` km/h.

    :param diffusion: Spreading of the raid, km^2/h (0 keeps it a tight column).
    :param dx: Grid spacing in km.
    """

    layers: tuple[Layer, ...]
    speed: float = 185.0
    diffusion: float = 50.0
    dx: float = 2.0

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("need at least one layer")
        if len({layer.name for layer in self.layers}) != len(self.layers):
            raise ValueError("layer names must be unique")
        if self.speed <= 0 or self.diffusion < 0 or self.dx <= 0:
            raise ValueError("speed and dx must be positive, diffusion non-negative")

    def raid(
        self,
        size: float,
        *,
        launch_window: float = 1.0,
        attacker_unit_cost: float = 0.0,
        magazines: Mapping[str, float] | None = None,
        t_max: float = 48.0,
    ) -> RaidResult:
        """Simulate one raid of ``size`` attackers entering the outer edge over
        ``launch_window`` hours (0 = all at once).

        :param magazines: Shots left per layer at the start (default: each layer's
            ``magazine``), for campaigns.
        """
        if size <= 0:
            raise ValueError("size must be positive")
        if launch_window < 0:
            raise ValueError("launch_window must be non-negative")
        mags = {layer.name: layer.magazine for layer in self.layers}
        if magazines:
            unknown = set(magazines) - set(mags)
            if unknown:
                raise ValueError(f"unknown layers: {', '.join(sorted(unknown))}")
            mags.update(magazines)
        return _simulate(self, float(size), launch_window, attacker_unit_cost, mags, t_max)

    def leakage_curve(
        self, sizes: ArrayLike, **raid_kwargs: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Leak fraction across raid sizes and the local elasticity of leakers with
        respect to raid size (``d log leaked / d log size``; 1 = proportional,
        above 1 = saturation, below 1 = economies of scale in defence)."""
        n = np.sort(np.asarray(sizes, float))
        if n.size < 2 or np.any(n <= 0):
            raise ValueError("need at least two positive sizes")
        leaked = np.array([self.raid(s, **raid_kwargs).leaked for s in n])  # type: ignore[arg-type]
        fraction = leaked / n
        elasticity = np.gradient(np.log(np.maximum(leaked, 1e-12)), np.log(n))
        return n, fraction, elasticity

    def campaign(
        self,
        sizes: Sequence[float],
        *,
        restock: Mapping[str, float] | None = None,
        launch_window: float = 1.0,
        attacker_unit_cost: float = 0.0,
    ) -> CampaignResult:
        """Night after night of raids against finite, slowly restocked magazines.

        :param restock: Shots added to each layer between raids (capped at its
            ``magazine``); layers not listed are not restocked.
        """
        restock = dict(restock or {})
        cap = {layer.name: layer.magazine for layer in self.layers}
        unknown = set(restock) - set(cap)
        if unknown:
            raise ValueError(f"unknown layers: {', '.join(sorted(unknown))}")
        mags = dict(cap)
        results = []
        stock = []
        for i, size in enumerate(sizes):
            if i:
                mags = {k: min(cap[k], v + restock.get(k, 0.0)) for k, v in mags.items()}
            res = self.raid(
                size,
                launch_window=launch_window,
                attacker_unit_cost=attacker_unit_cost,
                magazines=mags,
            )
            mags = {r.name: r.magazine_left for r in res.layers}
            results.append(res)
            stock.append([mags[layer.name] for layer in self.layers])
        return CampaignResult(
            tuple(results), tuple(layer.name for layer in self.layers), np.array(stock)
        )

    def strike(self, waves: Sequence[Wave], *, t_max: float = 12.0) -> StrikeResult:
        """Simulate a mixed strike: several waves (drones, cruise or ballistic missiles)
        with their own speeds and launch times, against layers that share fire capacity
        and magazines between them.

        A layer splits its shots across the waves it can engage in proportion to how
        many of each are inside it (a multi-prey Holling type II response). So drones
        arriving with missiles draw fire and use up interceptors the missiles would
        otherwise have faced.
        """
        state = _StrikeState(self, waves)
        while state.t < t_max and not state.done:
            state.step()
        return state.result()

    def with_surge(self, surge: float) -> AirDefense:
        """The same defence with every layer's ``surge`` set (see :class:`Layer`)."""
        return replace(self, layers=tuple(replace(lay, surge=surge) for lay in self.layers))


@dataclass(frozen=True)
class CampaignResult:
    raids: tuple[RaidResult, ...]
    layer_names: tuple[str, ...]
    magazines: NDArray[np.float64] = field(repr=False)
    """Shots left after each raid, one column per layer."""

    @property
    def leaked(self) -> NDArray[np.float64]:
        return np.array([r.leaked for r in self.raids])

    @property
    def leak_fraction(self) -> NDArray[np.float64]:
        return np.array([r.leak_fraction for r in self.raids])

    def empty_after(self, layer: str) -> int | None:
        """Index of the first raid after which ``layer`` had no shots left."""
        col = self.magazines[:, self.layer_names.index(layer)]
        hits = np.flatnonzero(col <= 1e-9)
        return int(hits[0]) if hits.size else None


def _simulate(
    defense: AirDefense,
    size: float,
    window: float,
    unit_cost: float,
    mags: dict[str, float],
    t_max: float,
) -> RaidResult:
    layers = defense.layers
    v, diff, dx = defense.speed, defense.diffusion, defense.dx
    extent = max(layer.outer for layer in layers) + 4 * dx
    n = math.ceil(extent / dx)
    centres = (np.arange(n) + 0.5) * dx
    masks = [(centres >= lay.inner) & (centres < lay.outer) for lay in layers]
    if not all(m.any() for m in masks):
        raise ValueError("a layer is narrower than the grid; reduce dx")
    # Largest stable explicit step: upwind advection is then an exact shift when
    # diffusion is 0, avoiding numerical smearing of the raid.
    dt = 1.0 / (v / dx + 2 * diff / (dx * dx))
    scale = [(size / lay.surge_reference) ** lay.surge for lay in layers]
    shots = np.zeros(len(layers))
    kills = np.zeros(len(layers))
    left = np.array([mags[lay.name] for lay in layers], float)

    u = np.zeros(n)  # attackers per km
    leaked = 0.0
    launched = 0.0
    times: list[float] = []
    rates: list[float] = []
    t = 0.0
    while t < t_max:
        # Launch into the outermost cell.
        if launched < size:
            add = size - launched if window == 0 else min(size - launched, size * dt / window)
            u[-1] += add / dx
            launched += add
        # Transport: upwind advection toward x = 0 and diffusion; leaks at x = 0.
        out = v * dt / dx * u
        flow = out.copy()
        new = u - out
        new[:-1] += flow[1:]
        step_leak = flow[0] * dx
        if diff > 0:
            k = diff * dt / (dx * dx)
            padded = np.r_[0.0, u, u[-1]]  # absorbing at the target, reflecting outside
            new += k * (padded[2:] - 2 * u + padded[:-2])
            step_leak += k * u[0] * dx
        u = np.maximum(new, 0.0)
        leaked += step_leak
        # Engagement, layer by layer.
        for i, (lay, mask) in enumerate(zip(layers, masks, strict=True)):
            present = float(u[mask].sum() * dx)
            if present <= 1e-12 or left[i] <= 0:
                continue
            f, e = lay.fire_rate * scale[i], lay.engage_rate_for(v) * scale[i]
            rate = e * present if math.isinf(f) else f * e * present / (f + e * present)
            fired = min(rate * dt, left[i])
            killed = min(lay.pk * fired, 0.999 * present)
            u[mask] *= 1 - killed / present
            shots[i] += fired
            kills[i] += killed
            left[i] -= fired
        t += dt
        times.append(t)
        rates.append(step_leak / dt)
        if launched >= size and u.sum() * dx < 1e-6 * size:
            break
    reports = tuple(
        LayerReport(
            lay.name,
            float(shots[i]),
            float(kills[i]),
            float(left[i]),
            float(shots[i] * lay.cost_per_shot),
        )
        for i, lay in enumerate(layers)
    )
    return RaidResult(size, leaked, reports, size * unit_cost, np.array(times), np.array(rates))


@dataclass(frozen=True)
class Wave:
    """One wave of a mixed strike.

    :param speed: km/h (drones ~185, cruise missiles ~800, ballistic missiles ~7,000
        averaged over the last few hundred km).
    :param launch_at: Hours after the start when the wave reaches the outer edge.
    :param launch_window: Hours over which the wave arrives there (0 = all at once).
    :param diffusion: Spreading, km^2/h.
    """

    name: str
    size: float
    speed: float
    launch_at: float = 0.0
    launch_window: float = 0.0
    diffusion: float = 0.0
    unit_cost: float = 0.0
    kind: str = ""
    """Class of attacker (e.g. "loitering drone"), for kill chances set by class."""

    def __post_init__(self) -> None:
        if self.size <= 0 or self.speed <= 0:
            raise ValueError(f"wave {self.name!r}: size and speed must be positive")
        if self.launch_at < 0 or self.launch_window < 0 or self.diffusion < 0:
            raise ValueError(f"wave {self.name!r}: times and diffusion must be non-negative")


@dataclass(frozen=True)
class StrikeResult:
    waves: tuple[str, ...]
    launched: NDArray[np.float64]
    leaked: NDArray[np.float64]
    layers: tuple[str, ...]
    kills: NDArray[np.float64]
    """Kills by layer (rows) and wave (columns)."""
    shots: NDArray[np.float64]
    """Shots by layer and wave."""
    magazine_left: NDArray[np.float64]

    def leak_fraction(self, wave: str) -> float:
        i = self.waves.index(wave)
        return float(self.leaked[i] / self.launched[i])

    def __str__(self) -> str:
        parts = [
            f"{w}: {lk:.1f} of {la:.0f} through ({100 * lk / la:.1f}%)"
            for w, la, lk in zip(self.waves, self.launched, self.leaked, strict=True)
        ]
        return "; ".join(parts)


class _StrikeState:
    """Mutable state of a strike simulation, advanced one step at a time."""

    def __init__(self, defense: AirDefense, waves: Sequence[Wave]) -> None:
        if not waves:
            raise ValueError("need at least one wave")
        if len({w.name for w in waves}) != len(waves):
            raise ValueError("wave names must be unique")
        self.defense, self.waves = defense, tuple(waves)
        layers, dx = defense.layers, defense.dx
        self.dx = dx
        self.n = math.ceil((max(lay.outer for lay in layers) + 4 * dx) / dx)
        self.extent = self.n * dx
        centres = (np.arange(self.n) + 0.5) * dx
        self.masks = [(centres >= lay.inner) & (centres < lay.outer) for lay in layers]
        if not all(m.any() for m in self.masks):
            raise ValueError("a layer is narrower than the grid; reduce dx")
        self.dt = min(1.0 / (w.speed / dx + 2 * w.diffusion / (dx * dx)) for w in waves)
        total = sum(w.size for w in waves)
        self.scale = np.array([(total / lay.surge_reference) ** lay.surge for lay in layers])
        self.pk = np.array([[lay.pk_for(w.name, w.kind) for w in waves] for lay in layers])
        self.engage = np.array([[lay.engage_rate_for(w.speed) for w in waves] for lay in layers])
        self.u = np.zeros((len(waves), self.n))
        self.launched = np.zeros(len(waves))
        self.leaked = np.zeros(len(waves))
        self.kills = np.zeros((len(layers), len(waves)))
        self.shots = np.zeros((len(layers), len(waves)))
        self.left = np.array([lay.magazine for lay in layers], float)
        self.hazard = np.zeros((len(layers), len(waves)))
        """Share of each wave inside each layer killed in the last step."""
        self.t = 0.0

    @property
    def done(self) -> bool:
        launched = np.allclose(self.launched, [w.size for w in self.waves])
        return bool(launched and self.u.sum() * self.dx < 1e-6 * sum(w.size for w in self.waves))

    def step(self) -> None:
        dt, dx, u = self.dt, self.dx, self.u
        for j, w in enumerate(self.waves):
            if self.launched[j] < w.size and self.t + dt > w.launch_at:
                if w.launch_window == 0:
                    add = w.size - self.launched[j]
                else:
                    add = min(w.size - self.launched[j], w.size * dt / w.launch_window)
                u[j, -1] += add / dx
                self.launched[j] += add
            out = w.speed * dt / dx * u[j]
            new = u[j] - out
            new[:-1] += out[1:]
            self.leaked[j] += out[0] * dx
            if w.diffusion > 0:
                k = w.diffusion * dt / (dx * dx)
                padded = np.r_[0.0, u[j], u[j, -1]]
                new += k * (padded[2:] - 2 * u[j] + padded[:-2])
                self.leaked[j] += k * u[j, 0] * dx
            u[j] = np.maximum(new, 0.0)
        self.hazard[:] = 0.0
        for i, lay in enumerate(self.defense.layers):
            if self.left[i] <= 0:
                continue
            present = u[:, self.masks[i]].sum(axis=1) * dx * (self.pk[i] > 0)
            if present.sum() <= 1e-12:
                continue
            # Multi-prey Holling type II: shots at wave j are F e_j N_j / (F + sum e_i N_i).
            f = lay.fire_rate * self.scale[i]
            demand = self.engage[i] * self.scale[i] * present
            total = demand.sum()
            rate = total if math.isinf(f) else f * total / (f + total)
            fired = min(rate * dt, self.left[i])
            share = fired * demand / total
            killed = np.minimum(self.pk[i] * share, 0.999 * present)
            frac = np.divide(killed, present, out=np.zeros_like(killed), where=present > 0)
            u[:, self.masks[i]] *= (1 - frac)[:, None]
            self.hazard[i] = frac
            self.shots[i] += share
            self.kills[i] += killed
            self.left[i] -= fired
        self.t += dt

    def result(self) -> StrikeResult:
        return StrikeResult(
            tuple(w.name for w in self.waves),
            self.launched.copy(),
            self.leaked.copy(),
            tuple(lay.name for lay in self.defense.layers),
            self.kills.copy(),
            self.shots.copy(),
            self.left.copy(),
        )


def _strike_steps(
    defense: AirDefense, waves: Sequence[Wave], *, t_max: float = 12.0
) -> Iterator[_StrikeState]:
    """Yield the strike's state after every step (for animation)."""
    state = _StrikeState(defense, waves)
    while state.t < t_max:
        state.step()
        yield state
        if state.done:
            break


def missile_defense() -> AirDefense:
    """An illustrative defence against a mixed strike of drones and ballistic missiles.

    Fighters far out engage only drones. An upper-tier missile defence engages only
    ballistic missiles. A lower-tier defence near the target engages both and shares
    its launchers and interceptors between them. Numbers are round placeholders, not
    fitted to any war: 100 missiles alone lose all but 6, but 150 drones sent ahead of
    them use up the lower tier and 33 missiles get through.
    """
    return AirDefense(
        (
            Layer("fighters", 300, 120, pk=0.6, fire_rate=150, engage_rate=2.0,
                  pk_against={"missiles": 0.0}, cost_per_shot=100_000),
            Layer("upper tier", 120, 40, pk=0.8, fire_rate=900, engage_rate=400,
                  magazine=70, pk_against={"drones": 0.0}, cost_per_shot=3_000_000),
            Layer("lower tier", 40, 0, pk=0.7, fire_rate=3000, engage_rate=600,
                  magazine=90, cost_per_shot=1_000_000),
        )
    )  # fmt: skip


def generic_defense(*, magazine_scale: float = 1.0) -> AirDefense:
    """An illustrative three-layer defence against one-way attack drones.

    Numbers are round, generic placeholders (not any country's order of battle):
    interceptor drones and aircraft far out, mobile gun teams in the middle, and
    surface-to-air missiles close in with a finite, expensive magazine.
    """
    s = magazine_scale
    return AirDefense(
        (
            Layer(
                "interceptors",
                300,
                150,
                pk=0.6,
                fire_rate=60,
                engage_rate=2.0,
                magazine=300 * s,
                cost_per_shot=5_000,
            ),
            Layer(
                "mobile guns", 150, 40, pk=0.3, fire_rate=100, engage_rate=2.0, cost_per_shot=500
            ),
            Layer(
                "missiles",
                40,
                0,
                pk=0.85,
                fire_rate=20,
                engage_rate=2.0,
                magazine=60 * s,
                cost_per_shot=500_000,
            ),
        )
    )


def ukraine_defense() -> AirDefense:
    """The three-layer defence fitted to Ukraine's nightly Shahed reports, Sep 2024 to
    Nov 2026 (``scripts/fit_ukraine_defense.py``).

    Starting from :func:`generic_defense`, the fit frees what the data can identify:
    a common scale on every layer's ``pk`` (x1.05) and ``surge`` (0.19, so each drone
    is engaged about 14% faster when the raid doubles). A fit with finite capacity ran
    capacity to the top of its range, so capacity is ``inf``: on a typical night no
    layer saturates. Magazines are unlimited within a night. Across seven raid-size
    groups the model misses the observed leak share by 1.5 percentage points on
    average; it overstates leakage on the smallest raids (24% vs 17%).

    The layer ranges, costs and the split of lethality between layers are not
    identified by these data (they only report launched and stopped per night) and stay
    at the generic values.
    """
    surge = 0.19
    return AirDefense(
        (
            Layer("interceptors", 300, 150, pk=0.63, fire_rate=math.inf, engage_rate=2.0,
                  cost_per_shot=5_000, surge=surge),
            Layer("mobile guns", 150, 40, pk=0.32, fire_rate=math.inf, engage_rate=2.0,
                  cost_per_shot=500, surge=surge),
            Layer("missiles", 40, 0, pk=0.90, fire_rate=math.inf, engage_rate=2.0,
                  cost_per_shot=500_000, surge=surge),
        )
    )  # fmt: skip


# --------------------------------------------------------------------- statistics


@dataclass(frozen=True)
class SaturationTest:
    """Within-period relationship between raid size and interception.

    ``slope`` is the change in log-odds that an attacker is stopped per unit of
    log raid size, holding the period (e.g. month) fixed, from a beta-binomial
    regression with one intercept per period. Fixed-capacity defences predict
    ``slope <= 0`` (see the module's proposition); positive values mean the
    defence did *better* against bigger raids.
    """

    slope: float
    se: float
    leak_elasticity: float
    """Within-period OLS slope of log leakers on log raid size (1 = proportional)."""
    leak_elasticity_se: float
    nights: int
    periods: int
    dispersion: float
    """Beta-binomial precision (smaller = more night-to-night variation)."""

    @property
    def interval(self) -> tuple[float, float]:
        return self.slope - 1.96 * self.se, self.slope + 1.96 * self.se

    @property
    def verdict(self) -> str:
        lo, hi = self.interval
        if lo > 0:
            return "interception improves with raid size: inconsistent with saturation"
        if hi < 0:
            return "interception worsens with raid size: consistent with saturation"
        return "no detectable relationship between raid size and interception"

    def __str__(self) -> str:
        return (
            f"slope {self.slope:+.3f} (95% CI {self.interval[0]:+.3f} to "
            f"{self.interval[1]:+.3f}); leak elasticity {self.leak_elasticity:.2f} "
            f"(se {self.leak_elasticity_se:.2f}); {self.nights} nights in {self.periods} "
            f"periods: {self.verdict}"
        )


def saturation_test(
    launched: ArrayLike, stopped: ArrayLike, periods: Sequence[object]
) -> SaturationTest:
    """Test whether interception falls (saturation) or rises with raid size.

    :param launched: Attackers launched each night.
    :param stopped: Attackers destroyed or otherwise stopped that night.
    :param periods: A label per night (e.g. ``"2025-10"``); comparisons are made
        only within a period, so slow changes in tactics and defences cancel.
    """
    n = np.asarray(launched, float)
    k = np.asarray(stopped, float)
    if n.shape != k.shape or n.ndim != 1 or len(periods) != n.size:
        raise ValueError("launched, stopped and periods must be 1-D and the same length")
    if np.any(n <= 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError("need launched > 0 and 0 <= stopped <= launched")
    k = np.round(k)
    n = np.round(n)
    labels, idx = np.unique(np.asarray([str(p) for p in periods]), return_inverse=True)
    if labels.size >= n.size - 1:
        raise ValueError("need several nights per period")
    logn = np.log(n)
    x = logn - logn.mean()
    p_count = labels.size
    # Periods with no size variation carry no information about the slope but are
    # kept: they inform the dispersion.

    def unpack(th: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        eta = th[0] * x + th[1 : 1 + p_count][idx]
        return eta, math.exp(th[-1])

    def nll_grad(th: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        eta, phi = unpack(th)
        p = np.clip(expit(eta), 1e-10, 1 - 1e-10)
        a, b = p * phi, (1 - p) * phi
        ll = betaln(k + a, n - k + b) - betaln(a, b)
        da = digamma(k + a) - digamma(a)
        db = digamma(n - k + b) - digamma(b)
        common = digamma(phi) - digamma(n + phi)
        dl_deta = phi * (da - db) * p * (1 - p)
        dl_dphi = p * da + (1 - p) * db + common
        grad = np.zeros_like(th)
        grad[0] = np.sum(dl_deta * x)
        grad[1 : 1 + p_count] = np.bincount(idx, dl_deta, minlength=p_count)
        grad[-1] = np.sum(dl_dphi) * phi
        return -float(ll.sum()), -grad

    share = np.clip(
        np.bincount(idx, k, minlength=p_count) / np.bincount(idx, n, minlength=p_count),
        0.02,
        0.98,
    )
    th0 = np.r_[0.0, np.log(share / (1 - share)), math.log(20.0)]
    fit = minimize(
        nll_grad, th0, jac=True, method="L-BFGS-B", bounds=[(-20, 20)] * (1 + p_count) + [(-5, 12)]
    )
    # Observed information from finite differences of the analytic gradient.
    h = 1e-5
    dim = fit.x.size
    hess = np.zeros((dim, dim))
    for j in range(dim):
        step = np.zeros(dim)
        step[j] = h
        hess[:, j] = (nll_grad(fit.x + step)[1] - nll_grad(fit.x - step)[1]) / (2 * h)
    hess = (hess + hess.T) / 2
    try:
        cov = np.linalg.pinv(hess)
        se = math.sqrt(max(cov[0, 0], 0.0))
    except np.linalg.LinAlgError:  # pragma: no cover
        se = math.nan

    # Within-period OLS of log leakers on log raid size.
    y = np.log(n - k + 0.5)
    yd = y - (np.bincount(idx, y) / np.bincount(idx))[idx]
    xd = logn - (np.bincount(idx, logn) / np.bincount(idx))[idx]
    sxx = float(xd @ xd)
    if sxx == 0:
        raise ValueError("raid sizes don't vary within any period")
    el = float(xd @ yd) / sxx
    resid = yd - el * xd
    dof = max(n.size - p_count - 1, 1)
    el_se = math.sqrt(float(resid @ resid) / dof / sxx)
    return SaturationTest(
        slope=float(fit.x[0]),
        se=se,
        leak_elasticity=el,
        leak_elasticity_se=el_se,
        nights=int(n.size),
        periods=int(p_count),
        dispersion=math.exp(float(fit.x[-1])),
    )
