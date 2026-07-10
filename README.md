# Statistics, by seeing it happen

**A ground-up probability & statistics course where almost every idea is _demonstrated on real data_ — not just asserted.**

🔗 **Live site: https://jd-jones-ases.github.io/seeing-statistics/**

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-009E73.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Built with Jupyter](https://img.shields.io/badge/built%20with-Jupyter%20%2B%20JupyterLite-4c72b0.svg)](https://jupyterlite.readthedocs.io/)
![Lessons](https://img.shields.io/badge/lessons-57%20notebooks-4c72b0.svg)

Every topic is taught on three levels at once — **the math** (the actual formula), **the meaning** (what the
number *is* and *does*), and **the interpretation** (what a result actually *says*, and what it doesn't). And
wherever possible you **watch it happen**: we treat a real dataset as the known truth and simulate — sampling it
hundreds of times so ideas like the Central Limit Theorem and the 95% confidence interval emerge in front of you.

## Three ways to use it

1. **Read it online** — open the [live site](https://jd-jones-ases.github.io/seeing-statistics/). Every chart is
   pre-drawn, the math is typeset, there's full-text search, dark mode, and it works offline once loaded.
2. **Run it live in your browser** — every lesson has a **Run live ▶** button that opens the real notebook in
   [JupyterLite](https://jupyterlite.readthedocs.io/): genuine Python (NumPy, pandas, Matplotlib, SciPy,
   **statsmodels**) running client-side, no install and no account. Change a sample size, a seed, or a confidence
   level and re-run.
3. **Run it on your own computer** — clone this repo and:
   ```bash
   python -m venv .venv && . .venv/Scripts/activate   # Windows; use .venv/bin/activate on macOS/Linux
   pip install -r requirements.txt
   python -m ipykernel install --user --name stats-course   # register the kernel the notebooks expect
   jupyter lab            # open the lessons/ folder
   ```
   The committed notebooks name a `stats-course` kernelspec; without that third line Jupyter will prompt
   for a kernel on every notebook (and the `tools/` build scripts can't re-execute at all).

## What makes it more than a slideshow

- **Interactive explorables** — drag a slider and watch the statistics move, computed live in your browser:
  the [100-confidence-interval coverage demo](https://jd-jones-ases.github.io/seeing-statistics/explore-confidence-intervals.html),
  a [Central Limit Theorem sampler](https://jd-jones-ases.github.io/seeing-statistics/explore-clt.html), and a
  [p-value / power explorer](https://jd-jones-ases.github.io/seeing-statistics/explore-p-value.html).
- **Two companions per lesson** — most data-driven lessons also offer the *same idea* **on another dataset** (see
  what changes) and a **"the code behind it"** companion that rebuilds the statistic from scratch in Python and
  checks it against the library. That's **57 notebooks** in all (21 core + 18 "another dataset" + 18 "code").
- **Self-teaching reference** — full-text search, per-lesson self-check quizzes, a glossary, and a printable
  formula cheat-sheet.
- **Honest by construction** — every dataset's source and licence is logged, charts are colourblind-safe, and each
  lesson keeps *the math*, *what the data shows*, and *what it means in the world* clearly separate.

## Course outline (two parts, at your pace)

**Part 1 — Seeing data & the logic of chance:** describing data · distributions, shape & outliers · correlation &
the regression line · sampling & study design · probability · counting · random variables · binomial & Poisson ·
the normal distribution · sampling distributions & the **CLT**.

**Part 2 — Drawing conclusions from data:** estimation & standard error · **confidence intervals (flagship)** ·
hypothesis testing, p-values, errors & power · inference for proportions · t-tests · ANOVA · chi-square ·
regression inference · multiple regression · the bootstrap & permutation tests.

## What's in here

| Folder / file | What it is |
|---|---|
| `lessons/` | The notebooks, in reading order. |
| `raw/` | Real datasets, downloaded once and kept **read-only**. See [`raw/SOURCES.md`](raw/SOURCES.md). |
| `lib/` | `statslab.py` — small shared helpers (data loading, the colourblind-safe chart palette). |
| `tools/` | The build: `lessonkit.py` (one MANIFEST drives everything) → `build_notebooks.py` → `render_html.py` → `make_index.py`, plus `build_jupyterlite.py`, `build_explorables.py`, and `verify_no_drift.py`. |
| `rendered/` | The static site (open `rendered/index.html`). |

## Building the site

The notebooks are committed pre-executed (so the published charts are the fact-checked ones). The site is
assembled — search index, the JupyterLite app, the landing page — and deployed to GitHub Pages by a GitHub
Actions workflow on every push; it does **not** re-execute the notebooks. To rebuild everything locally, see the
scripts in `tools/`.

## Licence & credits

© JD Jones, 2026. The course — text, code, charts, and site — is licensed
**[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**. The bundled **datasets are not** covered by
that licence; each keeps the licence of its original source (CC0 / public domain, OpenIntro educational use, the R
`datasets`/`HistData` packages, Gapminder, U.S. Government works such as NHANES and USGS, …). Full provenance and
licences: [`raw/SOURCES.md`](raw/SOURCES.md) and the site's
[Data sources & credits](https://jd-jones-ases.github.io/seeing-statistics/data-credits.html) page.

> For study and understanding — not professional (legal, medical, financial) advice.
