r"""
build_notebooks.py — generate every lesson notebook from its Python source.

Each lesson variant lives in tools/lessons_src/<module>.py and exposes a cells()
function. Reading order, filenames, and metadata all come from lessonkit.MANIFEST,
so this script just walks the manifest and builds whatever modules exist (handy
while new lessons/variants are still being written).

    ..\.venv\Scripts\python.exe tools\build_notebooks.py                 # build all
    ..\.venv\Scripts\python.exe tools\build_notebooks.py 03 relationships # build a few (by id)
    ..\.venv\Scripts\python.exe tools\build_notebooks.py 3               # or by core num

Writing .ipynb JSON by hand is error-prone; here each cell is a normal Python
string. A separate step (tools/render_html.py) executes them and exports HTML.
"""
import importlib
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # tools/
from lessonkit import MANIFEST, meta, save_lesson


def build(entry):
    mod_name = f"lessons_src.{entry['module']}"
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        print(f"  (skip) {entry['id']}: {mod_name} not written yet")
        return False
    importlib.reload(mod)
    save_lesson(entry["id"], mod.cells())
    return True


def _selected(argv):
    if not argv:
        return list(MANIFEST)
    chosen = []
    for a in argv:
        key = int(a) if a.isdigit() else a
        chosen.append(meta(key))
    return chosen


def main(argv):
    built = sum(build(e) for e in _selected(argv))
    print(f"done — built {built} notebook(s).")


if __name__ == "__main__":
    main(sys.argv[1:])
