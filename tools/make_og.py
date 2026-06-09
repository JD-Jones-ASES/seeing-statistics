r"""make_og.py — render the 1200x630 Open Graph social-preview card
(rendered/assets/og-card.png) shown when the site is shared. On-brand: a bell
curve and a confidence-interval motif over the course's blue.

    ..\.venv\Scripts\python.exe tools\make_og.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "rendered" / "assets" / "og-card.png"

NAVY = "#243044"
ACCENT = "#4c72b0"
TEAL = "#009E73"
VERM = "#D55E00"


def main():
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(NAVY)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(NAVY)
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.3); ax.axis("off")

    # bell curve motif, lower-right
    x = np.linspace(0, 12, 400)
    y = 1.55 * np.exp(-0.5 * ((x - 8.7) / 1.7) ** 2) + 0.35
    ax.plot(x, y, color="#9fb6da", lw=3, alpha=0.55)
    ax.fill_between(x, 0.35, y, color=ACCENT, alpha=0.18)

    # a few confidence-interval "caterpillar" lines (teal hits, one vermillion miss)
    rng = np.random.default_rng(7)
    for i in range(7):
        yy = 0.55 + i * 0.16
        c = VERM if i == 4 else TEAL
        x0 = 7.4 + rng.uniform(-0.5, 0.2); x1 = x0 + rng.uniform(1.6, 2.4)
        ax.plot([x0, x1], [yy, yy], color=c, lw=2.4, alpha=0.85,
                ls="--" if c == VERM else "-")

    ax.text(0.7, 4.9, "Statistics,", color="white", fontsize=58, fontweight="bold", va="center")
    ax.text(0.7, 3.95, "by seeing it happen", color="white", fontsize=58, fontweight="bold", va="center")
    ax.text(0.72, 2.95, "Probability & statistics — demonstrated on real data.",
            color="#cdd8ec", fontsize=24, va="center")
    ax.text(0.72, 2.35, "Read it · run the real Python live in your browser · play with it.",
            color="#cdd8ec", fontsize=20, va="center")
    ax.text(0.72, 1.35, "57 interactive lessons   ·   free & open   ·   CC BY-SA 4.0",
            color="#8ea3c7", fontsize=18, va="center")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=NAVY)
    plt.close(fig)
    print(f"wrote rendered/assets/{OUT.name}")


if __name__ == "__main__":
    main()
