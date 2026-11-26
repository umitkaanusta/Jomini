"""Shared drawing for the README's air-defence animations.

Each panel shows the sky from the target (left) out to the edge of the airspace, the
attackers flying in, one labelled bar per defending system showing where it fires, kills
flashing in that system's colour, and live counters of what got through and what the
defence has spent. Dots follow the model: each step, an attacker inside a system's band
is killed with the probability the model gives for that step.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jomini.airdefense import AirDefense, Wave, _StrikeState

W, PAD_L, PAD_R, SKY, BAR = 760, 165, 20, 120, 15
BG, SKY_BG, INK, MUTE, LEAK = (
    (250, 250, 247),
    (238, 240, 240),
    (28, 32, 38),
    (120, 126, 134),
    (200, 50, 43),
)
PALETTE = [
    (29, 122, 140),
    (178, 115, 23),
    (60, 140, 70),
    (98, 84, 170),
    (162, 58, 118),
    (190, 80, 50),
    (70, 110, 180),
    (120, 120, 40),
]
BALLISTIC, CRUISE = (214, 96, 22), (150, 110, 30)


def font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


@dataclass
class Panel:
    """One run of a strike against one defence.

    :param labels: Display label per layer name (defaults to the layer name).
    :param short: Short name per wave for the counters (defaults to the wave name).
    """

    title: str
    defense: AirDefense
    waves: Sequence[Wave]
    labels: dict[str, str] = field(default_factory=dict)
    short: dict[str, str] = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self) -> None:
        self.s = _StrikeState(self.defense, self.waves)
        self.rng = np.random.default_rng(self.seed)
        self.km = self.s.extent
        self.dots = [
            [j, math.nan, self.rng.uniform(0.06, 0.94), 0]
            for j, w in enumerate(self.s.waves)
            for _ in range(round(w.size))
        ]
        self.flash: list[list[float]] = []

    def spent(self) -> float:
        lays = self.s.defense.layers
        return float(sum(self.s.shots[i].sum() * lay.cost_per_shot for i, lay in enumerate(lays)))

    def fast_in_flight(self) -> bool:
        return any(d[3] == 1 and self.s.waves[d[0]].speed > 1000 for d in self.dots)

    def advance(self, until: float) -> None:
        s = self.s
        while s.t < until and not s.done:
            s.step()
            for j, w in enumerate(s.waves):
                due = int(s.launched[j] + 1e-9)
                mine = [d for d in self.dots if d[0] == j]
                for d in mine[:due]:
                    if d[3] == 0:
                        d[1], d[3] = self.km - 1, 1
                spread = math.sqrt(2 * w.diffusion * s.dt)
                for d in mine:
                    if d[3] != 1:
                        continue
                    d[1] -= w.speed * s.dt + spread * self.rng.standard_normal()
                    d[1] = min(d[1], 2 * self.km - d[1])
                    if d[1] <= 0:
                        d[3] = 3
                        self.flash.append([0.0, d[2], -1, 0])
                        continue
                    for i, lay in enumerate(s.defense.layers):
                        inside = lay.inner <= d[1] < lay.outer
                        if inside and self.rng.random() < s.hazard[i, j]:
                            d[3] = 2
                            self.flash.append([d[1], d[2], i, 0])
                            break


def render(
    path: str,
    panels: Sequence[Panel],
    *,
    busy: Callable[[float], bool] = lambda t: False,
    slow_step: float = 0.025,
    fast_step: float = 0.002,
    km_ticks: Sequence[float] = (0, 40, 150, 300),
    t_max: float = 6.0,
    fps: int = 15,
    hold: int = 40,
) -> None:
    """Animate the panels into a GIF. Time runs at ``slow_step`` hours per frame, and at
    ``fast_step`` while a fast attacker is in the air or ``busy(t)`` is true."""
    f_small, f_big = font(12), font(16)
    heights = [46 + SKY + 8 + BAR * len(p.defense.layers) + 10 for p in panels]
    H = sum(heights) + 24
    km = max(p.km for p in panels)

    def X(x: float) -> float:
        return PAD_L + x / km * (W - PAD_L - PAD_R)

    frames = []
    t = 0.0
    while True:
        fast = busy(t) or any(p.fast_in_flight() for p in panels)
        t += fast_step if fast else slow_step
        for p in panels:
            p.advance(t)
        img = Image.new("RGB", (W, H), BG)
        g = ImageDraw.Draw(img, "RGBA")
        y0 = 4
        for p, height in zip(panels, heights, strict=True):
            sky_top, sky_bot = y0 + 46, y0 + 46 + SKY

            def Y(v: float, top: int = sky_top) -> float:
                return top + 4 + v * (SKY - 8)

            g.rectangle([X(0), sky_top, X(p.km - 8), sky_bot], fill=SKY_BG)
            for dx_, h in ((0, 18), (8, 30), (16, 12)):
                g.rectangle([X(0) - 40 + dx_, sky_bot - h, X(0) - 34 + dx_, sky_bot], fill=INK)
            for j, x, y, state in p.dots:
                if state != 1:
                    continue
                cx, cy, w = X(x), Y(y), p.s.waves[j]
                if w.speed > 2000:
                    g.line([cx, cy, cx + 14, cy - 3], fill=BALLISTIC, width=3)
                elif w.speed > 400:
                    g.line([cx, cy, cx + 9, cy], fill=CRUISE, width=3)
                else:
                    g.rectangle([cx - 2, cy - 2, cx + 2, cy + 2], fill=INK)
            keep = []
            for fl in p.flash:
                x, y, i, age = fl
                a = max(0, 255 - age * 40)
                cx, cy = X(x), Y(y)
                if i < 0:
                    r = 3 + age * 1.5
                    g.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*LEAK, a))
                else:
                    c = (*PALETTE[int(i) % len(PALETTE)], a)
                    g.line([cx - 4, cy - 4, cx + 4, cy + 4], fill=c, width=2)
                    g.line([cx - 4, cy + 4, cx + 4, cy - 4], fill=c, width=2)
                fl[3] += 1
                if fl[3] < 7:
                    keep.append(fl)
            p.flash = keep
            for i, lay in enumerate(p.defense.layers):
                by = sky_bot + 8 + i * BAR
                colour = PALETTE[i % len(PALETTE)]
                g.rectangle([X(lay.inner), by + 3, X(lay.outer), by + BAR - 3], fill=colour)
                label = p.labels.get(lay.name, lay.name)
                g.text((PAD_L - 8 - g.textlength(label, font=f_small), by), label, fill=INK,
                       font=f_small)  # fmt: skip
            g.text((12, y0 + 4), p.title, fill=INK, font=f_big)
            parts = [
                f"{p.short.get(w.name, w.name)} through {lk:.0f}/{w.size:.0f}"
                for w, lk in zip(p.s.waves, p.s.leaked, strict=True)
            ]
            stats = "   ".join([*parts, f"spent ${p.spent() / 1e6:,.0f}M"])
            g.text((12, y0 + 24), stats, fill=LEAK, font=f_small)
            y0 += height
        for x in km_ticks:
            s = f"{x:g} km"
            g.text((X(x) - g.textlength(s, font=f_small) / 2, H - 20), s, fill=MUTE, font=f_small)
        frames.append(img.quantize(colors=64, dither=Image.Dither.NONE))
        settled = all(p.s.done for p in panels) and not any(p.flash for p in panels)
        if settled or t > t_max:
            break
    frames += [frames[-1]] * hold
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // fps,
                   loop=0, optimize=True)  # fmt: skip
    for p in panels:
        print(f"{p.title} | {p.s.result()} | spent ${p.spent() / 1e6:,.0f}M")
    print(len(frames), "frames ->", path)
