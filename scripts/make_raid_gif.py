"""Draw docs/raid.gif: a small and a large raid flying through the Ukraine-fitted defence.

uv run python scripts/make_raid_gif.py
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jomini.airdefense import ukraine_defense

W, LANE, TOP = 760, 170, 8
PAD_L, PAD_R = 70, 20
KM = 308.0
BG, INK, MUTE, LEAK = (250, 250, 247), (28, 32, 38), (120, 126, 134), (200, 50, 43)
RING = [(29, 122, 140), (178, 115, 23), (162, 58, 118)]
RAIDS = (30, 500)
FRAMES, HOLD, FPS = 150, 30, 15


class Raid:
    """Step the raid model forward and move dots that follow its kill rates."""

    def __init__(self, size: int, seed: int) -> None:
        d = ukraine_defense()
        self.layers, self.size, self.v, self.diff, self.dx = (
            d.layers,
            size,
            d.speed,
            d.diffusion,
            d.dx,
        )
        self.n = math.ceil((max(lay.outer for lay in self.layers) + 4 * self.dx) / self.dx)
        x = (np.arange(self.n) + 0.5) * self.dx
        self.masks = [(x >= lay.inner) & (x < lay.outer) for lay in self.layers]
        self.dt = 1 / (self.v / self.dx + 2 * self.diff / self.dx**2)
        self.scale = [(size / lay.surge_reference) ** lay.surge for lay in self.layers]
        self.u = np.zeros(self.n)
        self.t = self.launched = self.leaked = 0.0
        self.kills = 0.0
        self.rng = np.random.default_rng(seed)
        self.dots = min(size, 250)
        self.weight = size / self.dots
        self.x = np.full(self.dots, np.nan)
        self.y = self.rng.uniform(0.06, 0.94, self.dots)
        self.state = np.zeros(self.dots, int)  # 0 waiting, 1 flying, 2 killed, 3 through
        self.flash: list[list[float]] = []  # [x, y, ring, age]

    def step(self) -> None:
        dt, dx, u = self.dt, self.dx, self.u
        if self.launched < self.size:
            add = min(self.size - self.launched, self.size * dt)  # one-hour launch window
            u[-1] += add / dx
            self.launched += add
        out = self.v * dt / dx * u
        new = u - out
        new[:-1] += out[1:]
        self.leaked += out[0] * dx
        k = self.diff * dt / dx**2
        padded = np.r_[0.0, u, u[-1]]
        new += k * (padded[2:] - 2 * u + padded[:-2])
        self.leaked += k * u[0] * dx
        self.u = u = np.maximum(new, 0.0)
        frac = []
        for lay, mask, s in zip(self.layers, self.masks, self.scale, strict=True):
            present = u[mask].sum() * dx
            if present <= 1e-12:
                frac.append(0.0)
                continue
            killed = min(lay.pk * lay.engage_rate * s * present * dt, 0.999 * present)
            u[mask] *= 1 - killed / present
            self.kills += killed
            frac.append(killed / present)
        self.t += dt
        # Dots: launch, fly with the same drift and spread, die at the model's rates.
        due = min(self.dots, int(self.launched / self.weight + 1e-9))
        waiting = np.flatnonzero(self.state == 0)[: max(0, due - int((self.state > 0).sum()))]
        self.x[waiting], self.state[waiting] = KM - 2, 1
        fly = self.state == 1
        self.x[fly] -= self.v * dt + math.sqrt(2 * self.diff * dt) * self.rng.standard_normal(
            fly.sum()
        )
        self.x[fly] = np.where(self.x[fly] > KM, 2 * KM - self.x[fly], self.x[fly])
        through = fly & (self.x <= 0)
        self.state[through] = 3
        for i in np.flatnonzero(through):
            self.flash.append([0.0, self.y[i], -1, 0])
        for j, lay in enumerate(self.layers):
            inside = (self.state == 1) & (self.x >= lay.inner) & (self.x < lay.outer)
            hit = inside & (self.rng.random(self.dots) < frac[j])
            self.state[hit] = 2
            for i in np.flatnonzero(hit):
                self.flash.append([self.x[i], self.y[i], j, 0])


def font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def main() -> None:
    raids = [Raid(size, seed) for seed, size in enumerate(RAIDS)]
    end = 3.2  # hours of raid shown
    per_frame = end / FRAMES
    f_small, f_big = font(13), font(17)
    H = TOP + LANE * len(raids) + 26

    def X(km: float) -> float:
        return PAD_L + km / KM * (W - PAD_L - PAD_R)

    frames = []
    for f in range(FRAMES + HOLD):
        for r in raids:
            while r.t < min(f + 1, FRAMES) * per_frame:
                r.step()
        img = Image.new("RGB", (W, H), BG)
        g = ImageDraw.Draw(img, "RGBA")
        for lane, r in enumerate(raids):
            y0 = TOP + lane * LANE
            top, bot = y0 + 30, y0 + LANE - 14

            def Y(v: float, top: int = top, bot: int = bot) -> float:
                return top + v * (bot - top)

            for j, lay in enumerate(r.layers):
                g.rectangle([X(lay.inner), top, X(lay.outer), bot], fill=(*RING[j], 26))
                if lane == 0:
                    g.text((X(lay.inner) + 6, top + 4), lay.name, fill=RING[j], font=f_small)
            for dx_, h in ((0, 22), (9, 36), (18, 16)):  # the city
                g.rectangle([X(0) - 44 + dx_, bot - h, X(0) - 37 + dx_, bot], fill=INK)
            alive = r.state == 1
            for x, y in zip(r.x[alive], r.y[alive], strict=True):
                cx, cy = X(x), Y(y)
                g.rectangle([cx - 2, cy - 2, cx + 2, cy + 2], fill=INK)
            keep = []
            for fl in r.flash:
                x, y, j, age = fl
                a = max(0, 255 - age * 40)
                cx, cy = X(x), Y(y)
                if j < 0:
                    rad = 3 + age * 1.5
                    g.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(*LEAK, a))
                else:
                    c = (*RING[int(j)], a)
                    g.line([cx - 4, cy - 4, cx + 4, cy + 4], fill=c, width=2)
                    g.line([cx - 4, cy + 4, cx + 4, cy - 4], fill=c, width=2)
                fl[3] += 1
                if fl[3] < 7:
                    keep.append(fl)
            r.flash = keep
            share = r.leaked / r.size
            g.text((PAD_L - 60, y0 + 6), f"{r.size} drones", fill=INK, font=f_big)
            label = f"got through: {r.leaked:.0f} ({100 * share:.0f}%)"
            g.text(
                (W - PAD_R - g.textlength(label, font=f_big), y0 + 6), label, fill=LEAK, font=f_big
            )
        for km in (0, 40, 150, 300):
            t = f"{km} km"
            g.text((X(km) - g.textlength(t, font=f_small) / 2, H - 22), t, fill=MUTE, font=f_small)
        frames.append(img.quantize(colors=48, dither=Image.Dither.NONE))
    frames[0].save("docs/raid.gif", save_all=True, append_images=frames[1:],
                   duration=1000 // FPS, loop=0, optimize=True)  # fmt: skip
    for r in raids:
        print(f"{r.size} drones: {r.leaked:.1f} through ({100 * r.leaked / r.size:.1f}%)")


if __name__ == "__main__":
    main()
