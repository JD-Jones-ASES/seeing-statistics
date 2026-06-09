r"""render_html.py — execute the lesson notebooks and export polished, self-hosted HTML.

For each lesson it (1) runs every cell with the project's 'stats-course' kernel so
charts get embedded, (2) saves the executed .ipynb back in place, (3) exports a
static HTML copy into rendered/, and (4) runs inject_assets(): it rewrites nav
links to the sibling .html pages and upgrades the bare nbconvert page into a
branded, accessible, offline-capable one — a shared top bar, a footer (licence +
how-to-cite + data credits + "mark complete"), self-hosted KaTeX math (replacing
the CDN MathJax), and the shared course.css / course.js. One edit here re-skins
all 57 pages.

    ..\.venv\Scripts\python.exe tools\render_html.py                 # execute + export all
    ..\.venv\Scripts\python.exe tools\render_html.py 03 relationships  # a few (by id)
    ..\.venv\Scripts\python.exe tools\render_html.py --no-exec        # re-export only (fast; no re-run)

Use --no-exec after changing only this file / the assets, to re-skin every page
without re-executing Python (the executed notebooks on disk are reused as-is).

Datasets in raw/ are only ever read, never modified.
"""
import re
import sys
import html
import pathlib
from urllib.parse import quote

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # tools/
from lessonkit import MANIFEST, ROOT, LESSONS, meta, _position_label

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

RENDERED = ROOT / "rendered"
RENDERED.mkdir(exist_ok=True)

# --- site identity (the published GitHub Pages target) ---------------------
SITE_URL = "https://jd-jones-ases.github.io/seeing-statistics"
REPO_SLUG = "JD-Jones-ASES/seeing-statistics"
COURSE_NAME = "Statistics, by seeing it happen"

# CDN loaders nbconvert injects that we strip for an offline, self-hosted page.
_CDN_LOADERS = [
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe"> </script>',
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>',
]

# Longest filenames first so a core name (01-x.ipynb) can't partially rewrite a
# variant that embeds it (01-x~the-code.ipynb).
ALL_FILES = sorted((m["file"] for m in MANIFEST), key=len, reverse=True)


def rewrite_links(html_str: str) -> str:
    for f in ALL_FILES:
        html_str = html_str.replace(f'{f}"', f'{f[:-6]}.html"')   # foo.ipynb" -> foo.html"
    html_str = html_str.replace("../rendered/index.html", "index.html")
    return html_str


def _binder_url(entry):
    return f"https://mybinder.org/v2/gh/{REPO_SLUG}/HEAD?labpath={quote('lessons/' + entry['file'])}"


def _live_url(entry):
    # in-browser JupyterLite copy (built in the publish phase under rendered/live/)
    return f"live/lab/index.html?path={quote(entry['file'])}"


def _head_block(entry):
    title = html.escape(f"{entry['title']} — {COURSE_NAME}")
    desc = html.escape(entry.get("blurb", COURSE_NAME))
    return f"""
<meta name="description" content="{desc}"/>
<meta name="sc-lesson" content="{html.escape(entry['id'])}"/>
<meta name="theme-color" content="#4c72b0"/>
<meta property="og:type" content="article"/>
<meta property="og:site_name" content="{html.escape(COURSE_NAME)}"/>
<meta property="og:title" content="{html.escape(entry['title'])}"/>
<meta property="og:description" content="{desc}"/>
<meta property="og:image" content="{SITE_URL}/assets/og-card.png"/>
<meta property="og:url" content="{SITE_URL}/{entry['file'][:-6]}.html"/>
<meta name="twitter:card" content="summary_large_image"/>
<link rel="icon" href="assets/favicon.svg"/>
<link rel="stylesheet" href="assets/vendor/katex/katex.min.css"/>
<link rel="stylesheet" href="assets/course.css"/>
"""


def _topbar(entry):
    pos = html.escape(_position_label(entry))
    return f"""
<div class="sc-topbar">
  <a class="sc-home" href="index.html">{html.escape(COURSE_NAME)} <span>· course map</span></a>
  <span class="sc-spacer"></span>
  <span class="sc-pos" style="color:var(--sc-muted);font-size:12.5px">{pos}</span>
  <a class="sc-btn run" href="{_live_url(entry)}" title="Run this lesson live in your browser — no install">Run live ▶</a>
  <button class="sc-btn" type="button" onclick="scToggleTheme()" title="Toggle light / dark" aria-label="Toggle light or dark theme">◐</button>
</div>
"""


def _footer(entry):
    cite = (f"Jones, JD ({2026}). <em>{html.escape(entry['title'])}</em>. In "
            f"<em>{html.escape(COURSE_NAME)}</em>. {SITE_URL}/{entry['file'][:-6]}.html")
    return f"""
<footer class="sc-footer">
  <p class="sc-done"><label><input type="checkbox" id="sc-done-box"/> Mark this lesson complete
    <span style="font-size:12px">(saved on this device only)</span></label></p>
  <p><b>Run it yourself:</b>
     <a href="{_live_url(entry)}">in your browser (JupyterLite)</a> ·
     <a href="{_binder_url(entry)}">on Binder</a> — no install, no account.</p>
  <details class="sc-solution"><summary>How to cite this lesson</summary>
    <div class="sc-cite">{cite}</div></details>
  <p>© JD Jones, {2026}. Licensed <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.
     Bundled datasets keep their own licences — see <a href="data-credits.html">data sources &amp; credits</a>.
     For study and understanding, not professional advice.</p>
  <p><a href="index.html">↑ Course map</a> · <a href="glossary.html">Glossary</a> ·
     <a href="formula-sheet.html">Formula sheet</a> · <a href="data-credits.html">Data sources</a></p>
</footer>
"""


def _scripts():
    # self-hosted KaTeX, then course.js (which calls renderMathInElement on load).
    return """
<script defer src="assets/vendor/katex/katex.min.js"></script>
<script defer src="assets/vendor/katex/contrib/auto-render.min.js"></script>
<script defer src="assets/course.js"></script>
"""


def inject_assets(html_str: str, entry) -> str:
    """Turn a bare nbconvert page into the branded, accessible, offline course page."""
    # 1. real <title>
    html_str = html_str.replace("<title>Notebook</title>",
                                f"<title>{html.escape(entry['title'])} · {html.escape(COURSE_NAME)}</title>", 1)
    # 2. strip CDN loaders (math is self-hosted KaTeX; require.js is unused here)
    for loader in _CDN_LOADERS:
        html_str = html_str.replace(loader, "")
    # 3. head additions (before </head>)
    html_str = html_str.replace("</head>", _head_block(entry) + "</head>", 1)
    # 4. top bar right after <body ...>
    html_str = re.sub(r"(<body[^>]*>)", r"\1" + _topbar(entry), html_str, count=1)
    # 5. footer + scripts right before </body>
    html_str = html_str.replace("</body>", _footer(entry) + _scripts() + "</body>", 1)
    # 6. fix nav links to sibling .html
    return rewrite_links(html_str)


def render(entry, execute=True):
    nb_path = LESSONS / entry["file"]
    if not nb_path.exists():
        print(f"  (skip) {entry['id']}: {entry['file']} not built yet")
        return False

    nb = nbformat.read(nb_path, as_version=4)
    if execute:
        print(f"executing {entry['file']} ...")
        ep = ExecutePreprocessor(timeout=900, kernel_name="stats-course")
        ep.preprocess(nb, {"metadata": {"path": str(LESSONS)}})
        nbformat.write(nb, nb_path)   # save executed notebook (charts embedded)
    else:
        print(f"re-exporting {entry['file']} (no re-run) ...")

    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(nb)
    out = RENDERED / (entry["file"][:-6] + ".html")
    out.write_text(inject_assets(body, entry), encoding="utf-8")
    print(f"  wrote rendered/{out.name}")
    return True


def _selected(argv):
    if not argv:
        return list(MANIFEST)
    return [meta(int(a) if a.isdigit() else a) for a in argv]


def main(argv):
    execute = "--no-exec" not in argv
    argv = [a for a in argv if a != "--no-exec"]
    n = sum(render(e, execute=execute) for e in _selected(argv))
    print(f"done — {'rendered' if execute else 're-exported'} {n} lesson(s).")


if __name__ == "__main__":
    main(sys.argv[1:])
