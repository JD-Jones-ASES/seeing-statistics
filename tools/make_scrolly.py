r"""make_scrolly.py — build rendered/scrolly-confidence-intervals.html, a guided,
scroll-driven telling of the flagship 100-CI story (sticky canvas + prose steps,
driven by assets/scrolly.js over the embedded Ames population).

    ..\.venv\Scripts\python.exe tools\make_scrolly.py   (run after build_explorables.py)
"""
import pathlib
import pagekit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "rendered" / "scrolly-confidence-intervals.html"

STEPS = [
    ("Here is a whole real <b>population</b> — the living areas of <b>2,930 houses</b> in Ames, Iowa. "
     "Because we have all of them, we know the one thing you almost never know in real life: the "
     "<b>true mean μ</b> (the dashed line)."),
    ("In real life you can't measure everyone. So you take a <b>sample</b> — here, 50 houses (the ticks) — "
     "and from it you build a <b>95% confidence interval</b>: a range that says “the true mean is "
     "plausibly in here.” This one <b>caught μ</b>."),
    ("One interval is luck. The real claim of “95% confidence” is about the <b>long run</b>. So we "
     "repeat the whole process <b>100 times</b> — 100 samples, 100 intervals."),
    ("Count them: about <b>95 of 100</b> intervals contain the true mean (teal, solid). The rest "
     "<b>miss</b> (orange-red, dashed)."),
    ("Those misses aren't mistakes — they're the <b>~5% the method is allowed</b>. That is what "
     "“95% confidence” means: the <b>method's</b> long-run capture rate across many samples — "
     "<em>not</em> a 95% probability about any one interval you've already drawn."),
]


def main():
    steps_html = "".join(
        f'<div class="scrolly-step" data-step="{i}"><div class="box">{txt}</div></div>'
        for i, txt in enumerate(STEPS))
    body = f"""
<h1>What “95% confidence” really means</h1>
<p>Scroll through the story — the picture on the left rebuilds itself as you go. (Prefer to drive it yourself?
Open the <a href="explore-confidence-intervals.html">interactive version</a>.)</p>
<div class="scrolly">
  <div class="scrolly-graphic">
    <canvas id="scrolly-canvas" role="img"
      aria-label="The Ames house-size population with its true mean, then 100 confidence intervals of which about 95 capture that mean."></canvas>
  </div>
  <div class="scrolly-steps">{steps_html}</div>
</div>
<p class="explore-note">The same simulation, in editable Python, is the flagship lesson:
<a href="12-confidence-intervals.html">What a 95% confidence interval really means →</a> — or
<a href="explore-confidence-intervals.html">drag the sliders yourself</a>.</p>
"""
    extra_head = '<link rel="stylesheet" href="assets/explorables.css">'
    extra_scripts = ('<script defer src="assets/explorables-data.js"></script>'
                     '<script defer src="assets/scrolly.js"></script>')
    OUT.write_text(pagekit.page(
        "What 95% confidence really means",
        "A scroll-driven story: watch 100 confidence intervals form and ~95 capture the true mean.",
        body, label="scrollytelling story", extra_head=extra_head, extra_scripts=extra_scripts),
        encoding="utf-8")
    print(f"wrote rendered/{OUT.name}")


if __name__ == "__main__":
    main()
