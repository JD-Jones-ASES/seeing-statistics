"""
statslab.py — tiny shared helpers for the statistics course.

Two jobs:
  1. load_csv(...)  — load a dataset from raw/, downloading it ONCE if missing.
  2. use_course_style() — make every chart in the course look consistent.

Design rule (from the project's hard rules): datasets in raw/ are treated as
READ-ONLY "originals". We download a copy once, then only ever read it. Nothing
here ever edits or deletes a file in raw/.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

# raw/ sits next to lib/ , one level up from this file.
RAW_DIR = Path(__file__).resolve().parent.parent / "raw"


# ---------------------------------------------------------------------------
# Colorblind-safe palette (Okabe & Ito, 2008) + semantic color roles.
#
# Why this exists: the course originally used matplotlib's green (#2ca02c) and
# red (#d62728) — including to mean "captured vs. missed" on the flagship
# confidence-interval plot. A yellow-green/red pair is the classic red-green
# colour-vision-deficiency (CVD) trap, and it also collapses in grayscale print.
# The Okabe-Ito palette is engineered so every pair stays distinguishable for all
# common CVD types. Each old colour maps to a same-family Okabe-Ito colour
# (green->bluish-green, red->vermillion, orange->amber), so prose that says "the
# red curve" stays accurate while the chart becomes colourblind-safe.
#
# Charts where colour carries MEANING (hit vs. miss, reject vs. not) must ALSO
# add a non-colour cue — a marker shape or linestyle — so the distinction
# survives grayscale and severe CVD. The course's interactive web "explorables"
# import these same hex values so the demos match the notebooks exactly.
# ---------------------------------------------------------------------------
OKABE_ITO = {
    "black":      "#000000",
    "orange":     "#E69F00",   # amber
    "skyblue":    "#56B4E9",
    "green":      "#009E73",   # bluish green
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
}

# Semantic roles used across the lessons and the web explorables.
ACCENT  = "#4c72b0"                # the course's existing neutral blue accent (kept)
NEUTRAL = "#9aa7c7"                # the course's existing histogram fill (kept)
CAPTURE = OKABE_ITO["green"]       # a hit / captured / success / "good"
MISS    = OKABE_ITO["vermillion"]  # a miss / rejected / failure / "bad"

# An ordered, CVD-safe cycle for categorical series. Starts on the course accent
# so a default single-series chart looks unchanged.
CB_CYCLE = [ACCENT, OKABE_ITO["vermillion"], OKABE_ITO["green"], OKABE_ITO["orange"],
            OKABE_ITO["purple"], OKABE_ITO["skyblue"], OKABE_ITO["yellow"], OKABE_ITO["black"]]


def hit_miss_style(hit: bool) -> dict:
    """Marker + linestyle that distinguish hit/miss WITHOUT relying on colour.

    Returns kwargs to splat into a matplotlib plot call so a chart survives
    grayscale printing and severe colour-vision deficiency:
        ax.plot(..., color=(sl.CAPTURE if hit else sl.MISS), **sl.hit_miss_style(hit))
    """
    return {"marker": "o", "linestyle": "-"} if hit else {"marker": "X", "linestyle": "--"}


def load_csv(filename: str, url: str | None = None, **read_csv_kwargs) -> pd.DataFrame:
    """Return a dataset as a pandas DataFrame.

    Looks for ``raw/<filename>``. If it is not there and a ``url`` is given,
    download it once into raw/ (a cached, read-only copy), then load it. On
    later runs the local copy is used, so the course works fully offline.

    Example:
        df = load_csv("penguins.csv", url="https://.../penguins.csv")
    """
    RAW_DIR.mkdir(exist_ok=True)
    path = RAW_DIR / filename

    if not path.exists():
        if url is None:
            raise FileNotFoundError(
                f"'{path.name}' is not in raw/ and no download url was given.\n"
                f"Looked in: {RAW_DIR}"
            )
        print(f"Downloading {filename} ...")
        # A User-Agent header keeps picky servers from refusing the request.
        request = urllib.request.Request(url, headers={"User-Agent": "stats-course/1.0"})
        with urllib.request.urlopen(request) as response, open(path, "wb") as out:
            out.write(response.read())
        size_kb = path.stat().st_size / 1024
        print(f"Saved a local copy to raw/{filename}  ({size_kb:,.0f} KB). "
              f"It won't download again.")

    return pd.read_csv(path, **read_csv_kwargs)


def use_course_style() -> None:
    """Apply consistent, readable defaults to all charts in the course."""
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
        # "colorblind" is seaborn's built-in CVD-safe qualitative palette; it
        # makes hue-based and default-cycle charts colourblind-safe too.
        sns.set_theme(context="notebook", style="whitegrid", palette="colorblind")
    except Exception:
        pass

    from cycler import cycler
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 110,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "font.size": 11,
        # CVD-safe default colour cycle for any chart that doesn't set colours.
        "axes.prop_cycle": cycler(color=CB_CYCLE),
    })


def add_lib_to_path() -> None:
    """Convenience for notebooks: ensure this lib/ folder is importable.

    Most notebooks won't need to call this if they add the path themselves,
    but it's here so a lesson can simply do `import statslab` after a one-line
    sys.path tweak shown at the top of each notebook.
    """
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
