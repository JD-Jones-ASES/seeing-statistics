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
        sns.set_theme(context="notebook", style="whitegrid")
    except Exception:
        pass

    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 110,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "font.size": 11,
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
