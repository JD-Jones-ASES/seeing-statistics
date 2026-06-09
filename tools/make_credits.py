r"""make_credits.py — build rendered/data-credits.html from raw/SOURCES.md.

A friendly, public "Data sources & credits" page: every dataset, its origin,
license, and the lessons that use it. This is the public face of the project's
provenance log, and the place the CC BY-SA notice points to for the third-party
datasets it explicitly does NOT cover.

    ..\.venv\Scripts\python.exe tools\make_credits.py
"""
import html
import re
import pathlib

import pagekit

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = ROOT / "raw" / "SOURCES.md"
OUT = ROOT / "rendered" / "data-credits.html"


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _link_or_text(s):
    # the URL column is a bare URL; render it as a shortened link
    if s.startswith("http"):
        short = re.sub(r"^https?://", "", s)
        short = short[:48] + ("…" if len(short) > 48 else "")
        return f'<a href="{html.escape(s)}" rel="noopener">{html.escape(short)}</a>'
    return html.escape(s)


def parse_rows():
    rows = []
    in_table = False
    for line in SOURCES.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Dataset"):
            in_table = True
            continue
        if in_table and re.match(r"^\|[\s:|-]+\|?\s*$", line):
            continue  # the |---|---| separator
        if in_table and line.startswith("|"):
            c = _cells(line)
            if len(c) >= 6:
                rows.append({"file": c[0], "source": c[1], "url": c[2],
                             "license": c[3], "downloaded": c[4], "used": c[5]})
        elif in_table and not line.startswith("|"):
            break
    return rows


def main():
    rows = parse_rows()
    cards = []
    for r in rows:
        fname = html.escape(r["file"].replace("`", ""))
        cards.append(f"""
        <div class="credit">
          <h3>{fname}</h3>
          <p class="src">{html.escape(r['source'])}</p>
          <p><span class="lic">{html.escape(r['license'])}</span> · downloaded {html.escape(r['downloaded'])}</p>
          <p class="url">{_link_or_text(r['url'])}</p>
          <p class="used"><b>Used in:</b> {html.escape(r['used'])}</p>
        </div>""")

    body = f"""
<h1>Data sources &amp; credits</h1>
<p>This course makes claims from data, so it plays fair. Every dataset below was downloaded once from a reliable
public source, kept <strong>read-only</strong>, and logged here with its origin and licence — so any chart traces
back to its raw source.</p>
<p><strong>Licensing.</strong> The course itself (text, code, charts, site) is © JD Jones, licensed
<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>. The <strong>{len(rows)} datasets</strong>
listed here are <em>not</em> covered by that licence — each keeps the licence of its original source (CC0 / public
domain, OpenIntro educational use, R "datasets"/HistData, Gapminder CC-BY, U.S. Government public-domain works such
as NHANES and USGS, etc.). Reuse of a dataset follows its own terms; please credit the original source.</p>
<style>
.credit{{background:var(--sc-card);border:1px solid var(--sc-line);border-left:4px solid var(--sc-accent);
  border-radius:10px;padding:12px 16px;margin:12px 0}}
.credit h3{{margin:.1em 0 .3em;font-size:16px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.credit p{{margin:.15em 0;font-size:14px;color:var(--sc-muted)}}
.credit .src{{color:var(--sc-ink)}}
.credit .lic{{display:inline-block;background:rgba(0,158,115,.14);color:var(--sc-capture);
  border-radius:14px;padding:1px 10px;font-size:12.5px}}
.credit .used{{font-size:13px}}
</style>
<div class="credits">{''.join(cards)}</div>
<p style="margin-top:24px;color:var(--sc-muted);font-size:13.5px">A larger catalogue of verified, free,
no-account datasets — each mapped to the statistics concept it teaches best — lives in the project's
<code>raw/dataset-catalog.md</code>.</p>
"""
    OUT.write_text(pagekit.page("Data sources & credits",
                                "Every dataset in the course, its origin and licence, and the lessons that use it.",
                                body, label="data sources"), encoding="utf-8")
    print(f"wrote rendered/{OUT.name}  ({len(rows)} datasets)")


if __name__ == "__main__":
    main()
