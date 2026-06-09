r"""pagekit.py — shared chrome for the generated standalone pages (data credits,
glossary, formula sheet). Mirrors the lesson pages' top bar + footer so the whole
site feels like one place. Used by make_credits.py and make_reference.py."""

SITE = "Statistics, by seeing it happen"
SITE_URL = "https://jd-jones-ases.github.io/seeing-statistics"
REPO = "https://github.com/JD-Jones-ASES/seeing-statistics"


def page(title, desc, body, label="", math=False, print_btn=False):
    katex_head = ('<link rel="stylesheet" href="assets/vendor/katex/katex.min.css">' if math else "")
    katex_js = ('<script defer src="assets/vendor/katex/katex.min.js"></script>'
                '<script defer src="assets/vendor/katex/contrib/auto-render.min.js"></script>' if math else "")
    printb = ('<button class="sc-btn" type="button" onclick="window.print()" '
              'title="Print or save as PDF">Print</button>' if print_btn else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="{desc}">
<meta property="og:title" content="{title}"><meta property="og:description" content="{desc}">
<meta property="og:image" content="{SITE_URL}/assets/og-card.png">
<meta name="theme-color" content="#4c72b0">
<link rel="icon" href="assets/favicon.svg">
{katex_head}
<link rel="stylesheet" href="assets/course.css">
<title>{title} · {SITE}</title>
<script>(function(){{var t;try{{t=localStorage.getItem('sc-theme')}}catch(e){{}}if(t)document.documentElement.setAttribute('data-theme',t);}})();</script>
</head><body>
<div class="sc-topbar">
  <a class="sc-home" href="index.html">{SITE} <span>· course map</span></a>
  <span class="sc-spacer"></span>
  <span class="sc-pos" style="color:var(--sc-muted);font-size:12.5px">{label}</span>
  {printb}
  <button class="sc-btn" type="button" onclick="scToggleTheme()" aria-label="Toggle light or dark theme">◐</button>
</div>
<main class="explore-wrap">
{body}
</main>
<footer class="sc-footer">
  <p>© JD Jones, 2026 · <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>
  (bundled datasets keep their own licences — see <a href="data-credits.html">data sources</a>).
  For study and understanding, not professional advice.</p>
  <p><a href="index.html">↑ Course map</a> · <a href="glossary.html">Glossary</a> ·
  <a href="formula-sheet.html">Formula sheet</a> · <a href="data-credits.html">Data sources</a> ·
  <a href="{REPO}">Source on GitHub</a></p>
</footer>
<script defer src="assets/course.js"></script>
{katex_js}
</body></html>
"""
