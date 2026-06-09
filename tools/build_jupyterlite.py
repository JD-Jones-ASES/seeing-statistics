r"""build_jupyterlite.py — build the in-browser "Run live" sandbox (rendered/live).

Bundles the course's lesson notebooks (with outputs stripped, so visitors run
them fresh), the shared lib/ (statslab), and the raw/ datasets into the
JupyterLite filesystem. Because the CSVs ship inside the app, statslab.load_csv
finds them locally and never needs the network — the notebooks run unmodified.
Python itself runs in the browser via the Pyodide kernel (loaded from CDN).

    ..\.venv\Scripts\python.exe tools\build_jupyterlite.py

Output (rendered/live/) is a build artifact (gitignored); the deploy Action
rebuilds it. Datasets in raw/ are only ever read.
"""
import shutil
import subprocess
import sys
import pathlib

import nbformat
from nbconvert.preprocessors import ClearOutputPreprocessor

ROOT = pathlib.Path(__file__).resolve().parent.parent
LESSONS = ROOT / "lessons"
LIB = ROOT / "lib"
RAW = ROOT / "raw"
CONTENT = ROOT / "_live_content"          # transient staging dir
OUT = ROOT / "rendered" / "live"


def prepare_content():
    if CONTENT.exists():
        shutil.rmtree(CONTENT)
    (CONTENT / "lessons").mkdir(parents=True)
    clear = ClearOutputPreprocessor()
    n = 0
    for nb_path in sorted(LESSONS.glob("*.ipynb")):
        nb = nbformat.read(nb_path, as_version=4)
        clear.preprocess(nb, {})           # ship clean notebooks; visitor runs them
        nbformat.write(nb, CONTENT / "lessons" / nb_path.name)
        n += 1
    shutil.copytree(LIB, CONTENT / "lib", ignore=shutil.ignore_patterns("__pycache__"))
    # bundle the datasets (so load_csv reads them locally, offline). README files too.
    shutil.copytree(RAW, CONTENT / "raw", ignore=shutil.ignore_patterns("__pycache__"))
    print(f"staged {n} notebooks + lib + raw into {CONTENT.relative_to(ROOT)}")


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    exe = pathlib.Path(sys.executable).parent / (
        "jupyter-lite.exe" if sys.platform == "win32" else "jupyter-lite")
    # Pass the single staging dir so its children (lessons/, lib/, raw/) land at
    # the JupyterLite filesystem ROOT, mirroring the course layout — so the
    # notebooks' "find lib/" loop and statslab's raw/ path both resolve offline.
    cmd = [str(exe), "build",
           "--contents", str(CONTENT),
           "--output-dir", str(OUT)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    prepare_content()
    build()
    print(f"built JupyterLite at {OUT.relative_to(ROOT)}")
