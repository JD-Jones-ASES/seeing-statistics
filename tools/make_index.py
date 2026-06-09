r"""
make_index.py — build rendered/index.html, the course landing page / table of
contents. It links each lesson's read-anywhere HTML, shows what each teaches and
which dataset it uses, links any companion variants ("another dataset" / "the
code behind it"), and explains the two ways to use the course (run it live vs.
just read it). Generated from lessonkit.MANIFEST so it never drifts.

    ..\.venv\Scripts\python.exe tools\make_index.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # tools/
from lessonkit import (MANIFEST, ROOT, PART_NAMES, COURSE_TOTAL,
                       _lesson_pos, _variants_of, _VARIANT_TAG)

RENDERED = ROOT / "rendered"
RENDERED.mkdir(exist_ok=True)

CSS = """
:root{--ink:#1f2330;--muted:#5b6577;--line:#e3e7ef;--accent:#4c72b0;--bg:#f7f8fb;--card:#fff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:880px;margin:0 auto;padding:40px 22px 80px}
header h1{font-size:30px;margin:.1em 0 .15em}
header p.sub{color:var(--muted);margin:.2em 0 0;font-size:17px}
.how{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:18px 22px;margin:26px 0 8px}
.how h2{font-size:15px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 10px}
.how ul{margin:0;padding-left:20px}.how li{margin:.25em 0}
.how code{background:#eef1f7;padding:1px 6px;border-radius:5px;font-size:14px}
h2.sem{font-size:20px;margin:34px 0 6px;padding-top:10px;border-top:2px solid var(--line)}
h2.sem span{color:var(--muted);font-weight:400;font-size:15px}
.card{display:flex;gap:16px;align-items:flex-start;
  background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:16px 18px;margin:12px 0;transition:.12s}
.card:hover{border-color:var(--accent);box-shadow:0 3px 14px rgba(76,114,176,.13);transform:translateY(-1px)}
.card-main{display:flex;gap:16px;align-items:flex-start;text-decoration:none;color:inherit;flex:1}
.num{flex:0 0 46px;height:46px;border-radius:10px;background:var(--accent);color:#fff;
  display:grid;place-items:center;font-weight:700;font-size:18px}
.num.flag{background:#c98a2b}
.num.orient{background:#2ca02c}
.body h3{margin:.1em 0 .25em;font-size:18px}
.body p{margin:.15em 0;color:var(--muted);font-size:15px}
.badge{display:inline-block;margin-top:8px;font-size:12.5px;color:var(--accent);
  background:#eef2f9;border-radius:20px;padding:3px 11px}
.variants{margin-top:9px;font-size:13.5px}
.variants a{color:var(--accent);text-decoration:none;border:1px solid #d7deec;
  border-radius:16px;padding:2px 10px;margin-right:7px;white-space:nowrap}
.variants a:hover{background:#eef2f9}
.soon{opacity:.6}
footer{margin-top:40px;color:var(--muted);font-size:13.5px;text-align:center}
""".strip()


def _href(m):
    return m["file"][:-6] + ".html"


def card(m):
    is_flag = m.get("flagship", False)
    is_orient = m.get("orientation", False)
    cls = "flag" if is_flag else ("orient" if is_orient else "")
    label = "▶" if is_orient else ("★" if is_flag else str(_lesson_pos(m)))

    variant_html = ""
    extras = _variants_of(m)
    if extras:
        links = "".join(
            f'<a href="{_href(v)}">{_VARIANT_TAG[v["variant"]].capitalize()} →</a>'
            for v in extras)
        variant_html = f'<div class="variants">{links}</div>'

    return f"""
    <div class="card">
      <a class="card-main" href="{_href(m)}">
        <div class="num {cls}">{label}</div>
        <div class="body">
          <h3>{m['title']}</h3>
          <p>{m['blurb']}</p>
          <span class="badge">data: {m['data']}</span>
        </div>
      </a>
      {variant_html}
    </div>"""


def main():
    orientation = [m for m in MANIFEST if m.get("orientation")]
    part1 = [m for m in MANIFEST if m["variant"] == "core" and m["part"] == 1 and not m.get("orientation")]
    part2 = [m for m in MANIFEST if m["variant"] == "core" and m["part"] == 2]

    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             "<title>Statistics, by seeing it happen — course map</title>",
             f"<style>{CSS}</style></head><body><div class='wrap'>"]

    parts.append(
        "<header><h1>Statistics, by seeing it happen</h1>"
        "<p class='sub'>A ground-up probability &amp; statistics course where almost "
        "every idea is <em>demonstrated on real data</em> — not just asserted.</p></header>")

    parts.append(
        "<div class='how'><h2>Two ways to use this</h2><ul>"
        "<li><b>Run it (recommended):</b> double-click <code>start-jupyter.bat</code>, "
        "open the <code>lessons/</code> folder, and run each cell with <b>Shift+Enter</b>. "
        "Change a number, re-run, watch it change.</li>"
        "<li><b>Just read it:</b> click any lesson below to read the finished page — "
        "every chart already drawn, nothing to install.</li>"
        "<li>Each lesson teaches on three levels: <b>the math</b>, <b>what the number means</b>, "
        "and <b>what the result says about the world</b>. Many lessons also offer the same idea "
        "<b>on another dataset</b> and a <b>code-focused</b> companion.</li></ul></div>")

    if orientation:
        parts.append("<h2 class='sem'>New here? <span>· start with this — no statistics yet, "
                     "just how to drive a notebook</span></h2>")
        for m in orientation:
            parts.append(card(m))

    parts.append(f"<h2 class='sem'>Part 1 — {PART_NAMES[1]} "
                 f"<span>· lessons 1–{len(part1)}</span></h2>")
    for m in part1:
        parts.append(card(m))

    p2_start = len(part1) + 1
    parts.append(f"<h2 class='sem'>Part 2 — {PART_NAMES[2]} "
                 f"<span>· lessons {p2_start}–{COURSE_TOTAL}, the inference half</span></h2>")
    for m in part2:
        parts.append(card(m))

    parts.append("<footer>Everything runs locally and free. (These read-online pages need an "
                 "internet connection only to draw the math symbols; running the lessons in "
                 "JupyterLab needs nothing.) Dataset sources and licenses are logged in "
                 "<code>raw/SOURCES.md</code>. For study and understanding — not professional advice.</footer>")
    parts.append("</div></body></html>")

    out = RENDERED / "index.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
