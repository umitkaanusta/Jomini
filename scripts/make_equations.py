"""Draw the README's equations as SVG images (light and dark) into docs/eq/.

GitHub renders math in only one collapsed <details> block per page, so the README uses
images instead.

    uv run python scripts/make_equations.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EQUATIONS = {
    "raid": r"$\dfrac{\partial u}{\partial t} = v\,\dfrac{\partial u}{\partial x}"
    r" + D\,\dfrac{\partial^2 u}{\partial x^2}"
    r" - \sum_k h_k\,\mathbf{1}[x \in \mathrm{ring}\ k]\,u$",
    "rings": r"$N_k = \int_{\mathrm{ring}\ k} u\,dx, \qquad s_k = \dfrac{F_k\,e_k\,N_k}"
    r"{F_k + e_k\,N_k}, \qquad h_k = \dfrac{p_k\,s_k}{N_k}, \qquad \dfrac{dM_k}{dt} = -s_k$",
    "strike": r"$s_{kj} = \dfrac{F_k\,e_k\,N_{kj}}{F_k + e_k \sum_i N_{ki}}, \qquad"
    r" h_{kj} = \dfrac{p_{kj}\,s_{kj}}{N_{kj}}, \qquad \dfrac{dM_k}{dt} = -\sum_j s_{kj}$",
    "waterloo": r"$\dfrac{dR}{dt} = -\beta\,R\,B, \qquad \dfrac{dB}{dt} = -\rho\,R\,B$",
}
COLOURS = {"light": "#1f2328", "dark": "#e6edf3"}


def main() -> None:
    out = Path("docs/eq")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"mathtext.fontset": "cm", "svg.fonttype": "path"})
    for name, tex in EQUATIONS.items():
        for theme, colour in COLOURS.items():
            fig = plt.figure(figsize=(0.01, 0.01))
            fig.text(0, 0, tex, fontsize=17, color=colour)
            fig.savefig(out / f"{name}-{theme}.svg", transparent=True, bbox_inches="tight",
                        pad_inches=0.06)  # fmt: skip
            plt.close(fig)
    print("wrote", len(EQUATIONS) * len(COLOURS), "files to", out)


if __name__ == "__main__":
    main()
