r"""make_index.py — build rendered/index.html, the course landing page / table of
contents. It links each lesson's read-anywhere HTML, shows what each teaches and
which dataset it uses, links any companion variants ("another dataset" / "the
code behind it"), explains the ways to use the course, offers full-text search
(Pagefind, wired by the build), tracks per-device progress, and carries the
licence + credits footer. Generated from lessonkit.MANIFEST so it never drifts.

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
:root{--ink:#1f2330;--muted:#5b6577;--line:#e3e7ef;--accent:#4c72b0;--bg:#f7f8fb;--card:#fff;
  --capture:#009E73;--miss:#D55E00;}
:root[data-theme="dark"]{--ink:#e7eaf0;--muted:#9aa3b2;--line:#2b3142;--accent:#7aa2dd;--bg:#11141b;--card:#171b24;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ink:#e7eaf0;--muted:#9aa3b2;--line:#2b3142;--accent:#7aa2dd;--bg:#11141b;--card:#171b24;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
a{color:var(--accent)}
.wrap{max-width:880px;margin:0 auto;padding:28px 22px 80px}
.bar{display:flex;align-items:center;gap:12px;margin-bottom:8px}
.bar .grow{flex:1}
.tbtn{border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--accent);
  cursor:pointer;font-size:15px;padding:5px 10px}
.tbtn:hover{border-color:var(--accent)}
header h1{font-size:31px;margin:.1em 0 .15em}
header p.sub{color:var(--muted);margin:.2em 0 0;font-size:17px}
#search{margin:18px 0 6px}
#search .pagefind-ui__search-input{font-size:15px}
.progress{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin:16px 0;}
.progress .row{display:flex;align-items:center;gap:12px;font-size:14px;color:var(--muted)}
.track{flex:1;height:9px;background:var(--line);border-radius:6px;overflow:hidden}
.track > div{height:100%;width:0;background:var(--capture);transition:width .4s}
.progress .reset{border:none;background:none;color:var(--accent);cursor:pointer;font-size:13px}
.how{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin:18px 0 8px}
.how h2{font-size:15px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 10px}
.how ul{margin:0;padding-left:20px}.how li{margin:.3em 0}
.how code{background:var(--bg);padding:1px 6px;border-radius:5px;font-size:14px}
h2.sem{font-size:20px;margin:34px 0 6px;padding-top:10px;border-top:2px solid var(--line)}
h2.sem span{color:var(--muted);font-weight:400;font-size:15px}
.card{display:flex;gap:16px;align-items:flex-start;position:relative;
  background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:12px 0;transition:.12s}
.card:hover{border-color:var(--accent);box-shadow:0 3px 14px rgba(76,114,176,.13);transform:translateY(-1px)}
.card.is-done{border-color:var(--capture)}
.card.is-done::after{content:"✓ done";position:absolute;top:10px;right:14px;font-size:11.5px;color:var(--capture);
  border:1px solid var(--capture);border-radius:20px;padding:1px 8px}
.card-main{display:flex;gap:16px;align-items:flex-start;text-decoration:none;color:inherit;flex:1}
.num{flex:0 0 46px;height:46px;border-radius:10px;background:var(--accent);color:#fff;
  display:grid;place-items:center;font-weight:700;font-size:18px}
.num.flag{background:#c98a2b}
.num.orient{background:var(--capture)}
.body h3{margin:.1em 0 .25em;font-size:18px}
.body p{margin:.15em 0;color:var(--muted);font-size:15px}
.badge{display:inline-block;margin-top:8px;font-size:12.5px;color:var(--accent);
  background:rgba(76,114,176,.10);border-radius:20px;padding:3px 11px}
.variants{margin-top:9px;font-size:13.5px}
.variants a{color:var(--accent);text-decoration:none;border:1px solid var(--line);
  border-radius:16px;padding:2px 10px;margin-right:7px;white-space:nowrap}
.variants a:hover{background:rgba(76,114,176,.10)}
footer{margin-top:40px;color:var(--muted);font-size:13.5px;border-top:1px solid var(--line);padding-top:18px}
footer a{color:var(--accent)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
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

    numbered = "0" if is_orient else "1"
    return f"""
    <div class="card" data-lesson="{m['id']}" data-numbered="{numbered}">
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


SCRIPT = """
(function(){
  var t; try{ t=localStorage.getItem('sc-theme'); }catch(e){}
  if(t) document.documentElement.setAttribute('data-theme', t);
})();
function scToggleTheme(){
  var c=document.documentElement.getAttribute('data-theme')
        || (window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
  var n=c==='dark'?'light':'dark';
  document.documentElement.setAttribute('data-theme', n);
  try{ localStorage.setItem('sc-theme', n); }catch(e){}
}
document.addEventListener('DOMContentLoaded', function(){
  function refresh(){
    var done={}; try{ done=JSON.parse(localStorage.getItem('sc-progress')||'{}'); }catch(e){}
    var total=0, n=0;
    document.querySelectorAll('.card[data-numbered="1"]').forEach(function(c){
      total++;
      if(done[c.dataset.lesson]){ n++; c.classList.add('is-done'); } else { c.classList.remove('is-done'); }
    });
    var bar=document.getElementById('pbar'), lab=document.getElementById('plabel');
    if(bar) bar.style.width=(total? Math.round(100*n/total):0)+'%';
    if(lab) lab.textContent=n+' of '+total+' lessons marked complete';
  }
  refresh();
  var reset=document.getElementById('preset');
  if(reset) reset.addEventListener('click', function(){
    if(confirm('Clear your saved progress on this device?')){
      try{ localStorage.removeItem('sc-progress'); }catch(e){} refresh();
    }
  });
  // Pagefind full-text search (index built by the site build; absent in raw source)
  if(window.PagefindUI){ try{ new PagefindUI({element:'#search', showSubResults:true, resetStyles:false}); }catch(e){} }
});
"""


def main():
    orientation = [m for m in MANIFEST if m.get("orientation")]
    part1 = [m for m in MANIFEST if m["variant"] == "core" and m["part"] == 1 and not m.get("orientation")]
    part2 = [m for m in MANIFEST if m["variant"] == "core" and m["part"] == 2]

    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<meta name="description" content="A simulation-first probability and statistics course where almost every idea is demonstrated on real data. Read online, run live in your browser, or run locally.">',
             '<meta property="og:title" content="Statistics, by seeing it happen">',
             '<meta property="og:description" content="A ground-up probability and statistics course, demonstrated on real data. Read, run live in your browser, and tinker.">',
             '<meta property="og:image" content="https://jd-jones-ases.github.io/seeing-statistics/assets/og-card.png">',
             '<meta property="og:url" content="https://jd-jones-ases.github.io/seeing-statistics/">',
             '<meta name="theme-color" content="#4c72b0">',
             '<link rel="icon" href="assets/favicon.svg">',
             '<link rel="stylesheet" href="assets/pagefind/pagefind-ui.css">',
             "<title>Statistics, by seeing it happen — course map</title>",
             f"<style>{CSS}</style></head><body><div class='wrap'>"]

    parts.append(
        "<div class='bar'>"
        "<div class='grow'></div>"
        "<button class='tbtn' type='button' onclick='scToggleTheme()' title='Toggle light / dark' "
        "aria-label='Toggle light or dark theme'>◐</button></div>")

    parts.append(
        "<header><h1>Statistics, by seeing it happen</h1>"
        "<p class='sub'>A ground-up probability &amp; statistics course where almost "
        "every idea is <em>demonstrated on real data</em> — not just asserted.</p></header>")

    parts.append("<div id='search'></div>")

    parts.append(
        "<div class='progress'><div class='row'>"
        "<span id='plabel'>0 lessons marked complete</span>"
        "<div class='track'><div id='pbar'></div></div>"
        "<button class='reset' id='preset' type='button'>reset</button></div>"
        "<div class='row' style='margin-top:4px;font-size:12px'>Progress is saved on this device only — "
        "no account, nothing uploaded.</div></div>")

    parts.append(
        "<div class='how'><h2>Three ways to use this</h2><ul>"
        "<li><b>Run it live in your browser (no install):</b> open any lesson and click "
        "<b>Run live ▶</b> — the real Python runs in your browser, so you can change a sample size, "
        "a seed, or a confidence level and re-run to watch the result change.</li>"
        "<li><b>Just read it:</b> click any lesson below to read the finished page — every chart "
        "already drawn, the math typeset, nothing to install.</li>"
        "<li><b>Run it on your own computer:</b> install Python, then "
        "<code>pip install -r requirements.txt</code> and open the <code>lessons/</code> folder in "
        "JupyterLab.</li>"
        "<li>Each lesson teaches on three levels: <b>the math</b>, <b>what the number means</b>, and "
        "<b>what the result says about the world</b>. Most lessons also offer the same idea "
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

    parts.append(
        "<footer><p>© JD Jones, 2026. The course is licensed "
        "<a href='https://creativecommons.org/licenses/by-sa/4.0/'>CC BY-SA 4.0</a>; bundled datasets "
        "keep their own licences — see <a href='data-credits.html'>data sources &amp; credits</a>. "
        "For study and understanding, not professional advice.</p>"
        "<p><a href='glossary.html'>Glossary</a> · <a href='formula-sheet.html'>Formula sheet</a> · "
        "<a href='data-credits.html'>Data sources</a> · "
        "<a href='https://github.com/JD-Jones-ASES/seeing-statistics'>Source on GitHub</a></p>"
        "<p>Everything runs free; the reading pages work fully offline. "
        "Built with Python, Jupyter, and JupyterLite.</p></footer>")
    parts.append(f"<script>{SCRIPT}</script>")
    parts.append('<script src="assets/pagefind/pagefind-ui.js"></script>')
    parts.append("</div></body></html>")

    out = RENDERED / "index.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
