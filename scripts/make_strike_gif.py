"""Draw docs/strike.gif: ballistic missiles alone, then the same missiles behind drones.

uv run python scripts/make_strike_gif.py
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jomini.airdefense import Wave, _StrikeState, missile_defense

W, LANE, TOP = 760, 170, 8
PAD_L, PAD_R = 70, 20
BG, INK, MUTE, LEAK = (250, 250, 247), (28, 32, 38), (120, 126, 134), (200, 50, 43)
MISSILE = (214, 96, 22)
RING = [(29, 122, 140), (90, 84, 170), (162, 58, 118)]
FPS, HOLD = 15, 30


def missiles() -> Wave:
    return Wave("missiles", 100, 7000, launch_at=2.0, launch_window=0.1)


def drones() -> Wave:
    return Wave("drones", 150, 185, launch_window=1.0, diffusion=50)


class Panel:
    def __init__(self, title: str, waves: list[Wave], seed: int) -> None:
        self.title = title
        self.s = _StrikeState(missile_defense(), waves)
        self.rng = np.random.default_rng(seed)
        self.km = self.s.extent
        self.dots = []  # one per attacker: wave index, x, y, state
        for j, w in enumerate(self.s.waves):
            for _ in range(int(w.size)):
                self.dots.append([j, math.nan, self.rng.uniform(0.06, 0.94), 0])
        self.flash: list[list[float]] = []

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

    def in_flight(self, wave: int) -> bool:
        return any(d[0] == wave and d[3] == 1 for d in self.dots)


def font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def main() -> None:
    panels = [
        Panel("100 missiles", [missiles()], 0),
        Panel("150 drones, then 100 missiles", [missiles(), drones()], 1),
    ]
    f_small, f_big = font(13), font(17)
    H = TOP + LANE * 2 + 26
    km = panels[0].km

    def X(x: float) -> float:
        return PAD_L + x / km * (W - PAD_L - PAD_R)

    frames = []
    t = 0.0
    while True:
        busy = any(p.in_flight(0) for p in panels) or 1.95 < t < 2.2
        t += 0.002 if busy else 0.03
        for p in panels:
            p.advance(t)
        img = Image.new("RGB", (W, H), BG)
        g = ImageDraw.Draw(img, "RGBA")
        for lane, p in enumerate(panels):
            y0 = TOP + lane * LANE
            top, bot = y0 + 30, y0 + LANE - 14

            def Y(v: float, top: int = top, bot: int = bot) -> float:
                return top + v * (bot - top)

            for i, lay in enumerate(p.s.defense.layers):
                g.rectangle([X(lay.inner), top, X(lay.outer), bot], fill=(*RING[i], 26))
                if lane == 0:
                    g.text((X(lay.inner) + 6, top + 4), lay.name, fill=RING[i], font=f_small)
            for dx_, h in ((0, 22), (9, 36), (18, 16)):
                g.rectangle([X(0) - 44 + dx_, bot - h, X(0) - 37 + dx_, bot], fill=INK)
            for j, x, y, state in p.dots:
                if state != 1:
                    continue
                cx, cy = X(x), Y(y)
                if p.s.waves[j].name == "missiles":
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
                    c = (*RING[int(i)], a)
                    g.line([cx - 4, cy - 4, cx + 4, cy + 4], fill=c, width=2)
                    g.line([cx - 4, cy + 4, cx + 4, cy - 4], fill=c, width=2)
                fl[3] += 1
                if fl[3] < 7:
                    keep.append(fl)
            p.flash = keep
            g.text((PAD_L - 60, y0 + 6), p.title, fill=INK, font=f_big)
            m = p.s.waves.index(next(w for w in p.s.waves if w.name == "missiles"))
            label = f"missiles through: {p.s.leaked[m]:.0f} of 100"
            g.text((W - PAD_R - g.textlength(label, font=f_big), y0 + 6), label, fill=LEAK,
                   font=f_big)  # fmt: skip
        for x in (0, 40, 120, 300):
            s = f"{x} km"
            g.text((X(x) - g.textlength(s, font=f_small) / 2, H - 22), s, fill=MUTE, font=f_small)
        clock = f"{int(t)}:{int(t % 1 * 60):02d} after the first launch"
        g.text((X(210) - g.textlength(clock, font=f_small) / 2, H - 22), clock, fill=INK,
               font=f_small)  # fmt: skip
        frames.append(img.quantize(colors=48, dither=Image.Dither.NONE))
        if all(p.s.done for p in panels) and not any(p.flash for p in panels):
            break
        if t > 4:
            break
    frames += [frames[-1]] * HOLD
    frames[0].save("docs/strike.gif", save_all=True, append_images=frames[1:],
                   duration=1000 // FPS, loop=0, optimize=True)  # fmt: skip
    for p in panels:
        print(p.title, "|", p.s.result())
    print(len(frames), "frames")


if __name__ == "__main__":
    main()
