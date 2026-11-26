"""Draw docs/waterloo.gif: Waterloo as fought, and with Napoleon's army half as big again.

Each square is 1,000 men. Losses follow the attrition model fitted to the battle; each
army breaks at the median loss for its role in 1815 (from the breakpoint model fitted to
CDB90). The grids count which side won 100 refights with random breaking points.

    uv run python scripts/make_waterloo_gif.py
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jomini import Battle, LinearLaw, cdb90
from jomini.stochastic import default_breakpoints, monte_carlo

W, LANE, TOP = 760, 190, 6
BG, INK, MUTE = (250, 250, 247), (28, 32, 38), (120, 126, 134)
FRENCH, ALLIED, DEAD = (40, 78, 165), (180, 42, 50), (214, 210, 202)
SQ, GAP, ROWS = 7, 3, 12
FPS, FIGHT, RETREAT, HOLD = 15, 110, 25, 40
START_HOUR = 11.5  # the French guns opened around 11:30


def font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


class Army:
    def __init__(self, men: float, colour: tuple[int, int, int], x0: float, facing: int,
                 rng: np.random.Generator) -> None:  # fmt: skip
        self.n = round(men / 1000)
        self.colour, self.rng = colour, rng
        cols = -(-self.n // ROWS)
        self.pos = [
            (x0 + facing * (c * (SQ + GAP)), r * (SQ + GAP))
            for c in range(cols)
            for r in range(ROWS)
        ][: self.n]
        # Front ranks fall first, with some randomness.
        order = np.argsort([abs(p[0] - x0) + rng.uniform(0, 25) for p in self.pos])
        self.fall_order = list(order)
        self.dead: set[int] = set()

    def set_strength(self, men: float) -> None:
        target = self.n - round(men / 1000)
        while len(self.dead) < target:
            self.dead.add(int(self.fall_order[len(self.dead)]))


def main() -> None:
    waterloo = cdb90.get("Waterloo")
    model = waterloo.fit(LinearLaw)
    bp = default_breakpoints()
    breaks = (bp.median(1815, "attacker"), bp.median(1815, "defender"))
    allied = waterloo.defender.strength
    scenarios = []
    for k, title in ((1.0, "As fought"), (1.5, "Napoleon with half as many men again")):
        french = waterloo.attacker.strength * k
        battle = Battle(french, allied)
        fight = model.simulate(battle, breakpoint=breaks)
        mc = monte_carlo(model, battle, year=1815, seed=0)
        scenarios.append((title, french, fight, mc.p_red))
    f_small, f_big = font(13), font(16)
    H = TOP + LANE * 2
    frames = []
    rng = np.random.default_rng(3)
    lanes = []
    for title, french, fight, p in scenarios:
        fr = Army(french, FRENCH, 230, -1, rng)
        al = Army(allied, ALLIED, 290, 1, rng)
        wins = np.array([1] * round(100 * p) + [0] * (100 - round(100 * p)))
        rng.shuffle(wins)
        lanes.append((title, french, fight, p, fr, al, wins))
    total = FIGHT + RETREAT + HOLD
    for f in range(total):
        img = Image.new("RGB", (W, H), BG)
        g = ImageDraw.Draw(img)
        for lane, (title, french, fight, p, fr, al, wins) in enumerate(lanes):
            y0 = TOP + lane * LANE
            field_top = y0 + 46
            frac = min(f / FIGHT, 1.0)
            t = fight.duration * frac
            r_now, b_now = fight.at(t)
            fr.set_strength(r_now)
            al.set_strength(b_now)
            retreat = max(0, f - FIGHT) / RETREAT
            for army, shift in ((fr, -120 * min(retreat, 1.0)), (al, 0.0)):
                for i, (x, y) in enumerate(army.pos):
                    yy = field_top + y
                    if i in army.dead:
                        g.rectangle([x, yy, x + SQ, yy + SQ], fill=DEAD)
                    else:
                        xx = x + shift
                        if xx > -SQ:
                            g.rectangle([xx, yy, xx + SQ, yy + SQ], fill=army.colour)
            hour = START_HOUR + 24 * t
            clock = f"{int(hour):02d}:{int(hour % 1 * 60):02d}"
            g.text((12, y0 + 6), title, fill=INK, font=f_big)
            g.text((12, y0 + 26), f"French {french:,.0f}", fill=FRENCH, font=f_small)
            al_label = f"British, Dutch and Prussians {allied:,.0f}"
            g.text((520 - g.textlength(al_label, font=f_small), y0 + 26), al_label, fill=ALLIED,
                   font=f_small)  # fmt: skip
            status = clock if f < FIGHT else f"{clock}: the French army breaks"
            g.text((250 - g.textlength(status, font=f_small) / 2, y0 + LANE - 18), status,
                   fill=INK, font=f_small)  # fmt: skip
            # Tally of 100 refights.
            shown = min(100, int(100 * f / (FIGHT + RETREAT)))
            tx, ty = 572, field_top + 6
            for k in range(100):
                cx, cy = tx + (k % 10) * 11, ty + (k // 10) * 11
                colour = (FRENCH if wins[k] else ALLIED) if k < shown else DEAD
                g.rectangle([cx, cy, cx + 8, cy + 8], fill=colour)
            g.text((tx, y0 + 26), "100 refights", fill=MUTE, font=f_small)
            if shown == 100:
                g.text(
                    (tx, ty + 114),
                    f"Napoleon wins {round(100 * p)} of 100",
                    fill=FRENCH,
                    font=f_big,
                )
        frames.append(img.quantize(colors=16, dither=Image.Dither.NONE))
    frames[0].save("docs/waterloo.gif", save_all=True, append_images=frames[1:],
                   duration=1000 // FPS, loop=0, optimize=True)  # fmt: skip
    for title, french, fight, p, *_ in lanes:
        r, b = fight.remaining
        hours = 24 * fight.duration
        print(f"{title}: French {french:,.0f} -> {r:,.0f}, allies -> {b:,.0f}; "
              f"breaks after {hours:.1f} h; Napoleon wins {100 * p:.0f}%")  # fmt: skip


if __name__ == "__main__":
    main()
