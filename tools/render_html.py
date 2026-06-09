r"""
render_html.py — execute the lesson notebooks and export read-anywhere HTML.

For each lesson it: (1) runs every cell with the project's 'stats-course' kernel
so charts get embedded, (2) saves the executed .ipynb back in place, and
(3) exports a static HTML copy into rendered/ — rewriting the in-notebook
navigation links (which point at sibling .ipynb files, ideal inside JupyterLab)
to point at the sibling .html files instead, so Previous/Next/Course-map all
work in a plain browser too.

    ..\.venv\Scripts\python.exe tools\render_html.py                   # all lessons
    ..\.venv\Scripts\python.exe tools\render_html.py 03 relationships  # a few (by id)
    ..\.venv\Scripts\python.exe tools\render_html.py 3                 # or by core num

Datasets in raw/ are only ever read, never modified.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # tools/
from lessonkit import MANIFEST, ROOT, LESSONS, meta

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

RENDERED = ROOT / "rendered"
RENDERED.mkdir(exist_ok=True)

# Longest filenames first so a core name (01-x.ipynb) can't partially rewrite a
# variant that embeds it (01-x__the-code.ipynb).
ALL_FILES = sorted((m["file"] for m in MANIFEST), key=len, reverse=True)


def rewrite_links(html: str) -> str:
    # nav links target sibling notebooks; in the browser they should open the HTML.
    for f in ALL_FILES:
        html = html.replace(f'{f}"', f'{f[:-6]}.html"')   # foo.ipynb" -> foo.html"
    # the course-map link is ../rendered/index.html from a notebook; flat here.
    html = html.replace("../rendered/index.html", "index.html")
    return html


def render(entry):
    nb_path = LESSONS / entry["file"]
    if not nb_path.exists():
        print(f"  (skip) {entry['id']}: {entry['file']} not built yet")
        return False

    nb = nbformat.read(nb_path, as_version=4)
    print(f"executing {entry['file']} ...")
    ep = ExecutePreprocessor(timeout=900, kernel_name="stats-course")
    # run as if launched from lessons/, so the in-notebook lib/data paths resolve
    ep.preprocess(nb, {"metadata": {"path": str(LESSONS)}})
    nbformat.write(nb, nb_path)   # save executed notebook (charts embedded)

    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(nb)
    out = RENDERED / (entry["file"][:-6] + ".html")
    out.write_text(rewrite_links(body), encoding="utf-8")
    print(f"  wrote rendered/{out.name}")
    return True


def _selected(argv):
    if not argv:
        return list(MANIFEST)
    return [meta(int(a) if a.isdigit() else a) for a in argv]


def main(argv):
    n = sum(render(e) for e in _selected(argv))
    print(f"done — rendered {n} lesson(s).")


if __name__ == "__main__":
    main(sys.argv[1:])
