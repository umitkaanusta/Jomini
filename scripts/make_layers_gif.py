"""Draw docs/cheap-layers.gif: the same strike against missile batteries alone, and
against the same batteries behind cheap guns and interceptor drones.

    uv run python scripts/make_layers_gif.py
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jomini import equipment as eq
from jomini.airdefense import AirDefense, _StrikeState

W, PAD_L, PAD_R = 760, 150, 20
SKY, BAR = 120, 15
BG, INK, MUTE, LEAK = (250, 250, 247), (28, 32, 38), (120, 126, 134), (200, 50, 43)
MISSILE = (214, 96, 22)
COLOURS = {
    "NASAMS": (98, 84, 170),
    "Patriot PAC-3": (162, 58, 118),
    "Gepard": (178, 115, 23),
    "mobile gun team": (29, 122, 140),
    "interceptor drone team": (60, 140, 70),
}
LABELS = {
    "NASAMS": "NASAMS",
    "Patriot PAC-3": "Patriot",
    "Gepard": "Gepard",
    "mobile gun team": "gun teams",
    "interceptor drone team": "interceptor drones",
}
FPS, HOLD = 15, 40


def strike() -> list:
    return [
        eq.wave("Shahed-136", 100, launch_window=1.5),
        eq.wave("Iskander-M", 10, launch_at=2.5, launch_window=0.05),
    ]


def batteries() -> list:
    # Without cheaper layers, the missile batteries have to fire at the drones too.
    return [eq.layer("NASAMS", targets="all"), eq.layer("Patriot PAC-3", targets="all")]


CHEAP = {"Gepard": 6, "mobile gun team": 30, "interceptor drone team": 10}


def font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


class Panel:
    def __init__(self, title: str, defense: AirDefense, seed: int) -> None:
        self.title = title
        self.s = _StrikeState(defense, strike())
        self.rng = np.random.default_rng(seed)
        self.km = self.s.extent
        self.dots = [
            [j, math.nan, self.rng.uniform(0.06, 0.94), 0]
            for j, w in enumerate(self.s.waves)
            for _ in range(int(w.size))
        ]
        self.flash: list[list[float]] = []
        self.base = [lay.name.split(" x")[0] for lay in self.s.defense.layers]

    def spent(self) -> float:
        lays = self.s.defense.layers
        return float(sum(self.s.shots[i].sum() * lay.cost_per_shot for i, lay in enumerate(lays)))

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

    def flying(self, wave: int) -> bool:
        return any(d[0] == wave and d[3] == 1 for d in self.dots)


def main() -> None:
    cheap = AirDefense((*batteries(), *(eq.layer(n, c) for n, c in CHEAP.items())), edge=300)
    panels = [
        Panel("Missile batteries only", AirDefense(tuple(batteries()), edge=300), 0),
        Panel("The same, behind cheap guns and drones", cheap, 1),
    ]
    f_small, f_big = font(12), font(16)
    heights = [34 + SKY + 8 + BAR * len(p.base) + 10 for p in panels]
    H = sum(heights) + 24
    km = panels[0].km

    def X(x: float) -> float:
        return PAD_L + x / km * (W - PAD_L - PAD_R)

    frames = []
    t = 0.0
    while True:
        busy = any(p.flying(1) for p in panels) or 2.45 < t < 2.6
        t += 0.002 if busy else 0.025
        for p in panels:
            p.advance(t)
        img = Image.new("RGB", (W, H), BG)
        g = ImageDraw.Draw(img, "RGBA")
        y0 = 4
        for p, height in zip(panels, heights, strict=True):
            sky_top, sky_bot = y0 + 34, y0 + 34 + SKY

            def Y(v: float, top: int = sky_top) -> float:
                return top + 4 + v * (SKY - 8)

            g.rectangle([X(0), sky_top, X(300), sky_bot], fill=(238, 240, 240))
            for dx_, h in ((0, 18), (8, 30), (16, 12)):
                g.rectangle([X(0) - 40 + dx_, sky_bot - h, X(0) - 34 + dx_, sky_bot], fill=INK)
            for j, x, y, state in p.dots:
                if state != 1:
                    continue
                cx, cy = X(x), Y(y)
                if p.s.waves[j].kind == "ballistic missile":
                    g.line([cx, cy, cx + 14, cy - 3], fill=MISSILE, width=3)
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
                    c = (*COLOURS[p.base[int(i)]], a)
                    g.line([cx - 4, cy - 4, cx + 4, cy + 4], fill=c, width=2)
                    g.line([cx - 4, cy + 4, cx + 4, cy - 4], fill=c, width=2)
                fl[3] += 1
                if fl[3] < 7:
                    keep.append(fl)
            p.flash = keep
            # One bar per system, showing where it can fire.
            for i, (lay, base) in enumerate(zip(p.s.defense.layers, p.base, strict=True)):
                by = sky_bot + 8 + i * BAR
                g.rectangle([X(lay.inner), by + 3, X(lay.outer), by + BAR - 3], fill=COLOURS[base])
                label = LABELS[base] + (lay.name[len(base) :] if lay.name != base else "")
                g.text((PAD_L - 8 - g.textlength(label, font=f_small), by), label, fill=INK,
                       font=f_small)  # fmt: skip
            g.text((12, y0 + 4), p.title, fill=INK, font=f_big)
            drones, missiles = p.s.leaked
            stats = (f"drones through {drones:.0f}/100   Iskanders through {missiles:.0f}/10   "
                     f"spent ${p.spent() / 1e6:.0f}M")  # fmt: skip
            g.text((W - PAD_R - g.textlength(stats, font=f_small), y0 + 8), stats, fill=LEAK,
                   font=f_small)  # fmt: skip
            y0 += height
        for x in (0, 40, 60, 150, 300):
            s = f"{x} km"
            g.text((X(x) - g.textlength(s, font=f_small) / 2, H - 20), s, fill=MUTE, font=f_small)
        frames.append(img.quantize(colors=64, dither=Image.Dither.NONE))
        if (all(p.s.done for p in panels) and not any(p.flash for p in panels)) or t > 4:
            break
    frames += [frames[-1]] * HOLD
    frames[0].save("docs/cheap-layers.gif", save_all=True, append_images=frames[1:],
                   duration=1000 // FPS, loop=0, optimize=True)  # fmt: skip
    for p in panels:
        print(p.title, "|", p.s.result(), f"| spent ${p.spent() / 1e6:.0f}M")
    print(len(frames), "frames")


if __name__ == "__main__":
    main()
