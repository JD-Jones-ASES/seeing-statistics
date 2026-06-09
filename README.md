# Statistics, by seeing it happen

A self-paced course in probability & statistics — from the ground floor up —
where almost every idea is **demonstrated on real data**, not just asserted.

Every topic is taught on three levels at once:

1. **The math** — the actual formula and how to compute it.
2. **The meaning** — what the number *is* and what it *does*.
3. **The interpretation** — what a result actually *says* (and what it doesn't).

And wherever possible, you **watch it happen**: we take a real dataset, treat it
as the known truth, and simulate — sampling it hundreds of times to see ideas
like the Central Limit Theorem and the 95% confidence interval emerge in front
of you. You can change a number and re-run to see what changes.

---

## How to use it

1. **Just reading?** Double-click `view-lessons.bat` to open the **course map** in
   your browser — a table of contents linking every finished lesson, all charts
   already drawn, nothing to install.
2. **Want to run and tinker?** Double-click `start-jupyter.bat`. Your browser opens
   to JupyterLab; the lessons are in the `lessons/` folder, in order. (A black window
   stays open while it runs — closing it stops the course.)
3. **Work through a lesson:** read the text, run each cell (Shift+Enter), study
   the chart, then *tinker* — change a sample size or a confidence level and
   re-run to see the effect.
4. **Stop anytime.** Everything is saved as files on your computer. Nothing is
   online; nothing needs an account.

If `start-jupyter.bat` doesn't open, you can also launch from a terminal in this
folder with: `..\.venv\Scripts\jupyter-lab.exe`

## What's in here

| Folder / file | What it is |
|---|---|
| `lessons/` | The notebooks, in order. This is where you spend your time. |
| `raw/` | Real datasets, downloaded once and kept **read-only**. See `raw/SOURCES.md`. |
| `lib/` | Small shared helpers (`statslab.py`) for loading data and styling charts. |
| `tools/` | Scripts that generate, run, and render the lessons. You don't need these to use the course. |
| `rendered/` | Read-anywhere web (HTML) copies of every lesson. **Open `rendered/index.html` for the course map** — a table of contents that links them all. |
| `brief.md` | The one-page plan for this whole project. |
| `requirements.txt` | The list of libraries the course uses. |

## Course outline (two parts, at your pace)

**Part 1 — Seeing data & the logic of chance** — ✅ *all built*
0. ✅ **How these notebooks work (start here)** — a five-minute, no-statistics tour for anyone new to notebooks · *orientation*
1. ✅ Describing a real dataset — center, spread, shape, z-scores · *Palmer Penguins*
2. ✅ Distributions: shape, skew & outliers · *Old Faithful, Lending Club*
3. ✅ Relationships: correlation & the regression line · *Ames housing (+ penguins)*
4. ✅ Where data comes from: sampling & study design · *simulation on Ames*
5. ✅ Probability & conditional probability · *Titanic*
6. ✅ Counting: permutations & combinations · *dice & cards (simulation)*
7. ✅ Random variables, expectation & variance · *dice, US Births 2014*
8. ✅ Binomial & Poisson (counts) · *US Births 2014, USGS earthquakes*
9. ✅ The normal distribution & the empirical rule · *Galton family heights*
10. ✅ Sampling distributions & the **Central Limit Theorem** · *Lending Club + simulation*

**Part 2 — Drawing conclusions from data** — ✅ *all built*
11. ✅ Estimation & standard error · *Ames housing*
12. ✅ **Confidence intervals** — the flagship coverage simulation · *Ames housing*
13. ✅ Hypothesis testing: p-values, errors & power · *simulation*
14. ✅ Inference for proportions (one & two) · *US Births 2014*
15. ✅ t-tests: one-sample, two-sample & paired · *ToothGrowth, Swim*
16. ✅ Comparing many groups: ANOVA · *Palmer Penguins*
17. ✅ Chi-square: goodness-of-fit & independence · *Titanic*
18. ✅ Correlation & regression: inference · *Ames housing*
19. ✅ Multiple regression (intro) · *Ames housing*
20. ✅ The bootstrap & permutation tests · *Old Faithful, ToothGrowth*

> The order is built for *intuition*, not to match any official syllabus.
>
> **Two companions per lesson.** Every data-driven lesson also offers the *same idea*
> **on another dataset** (see what changes) and a **"the code behind it"** companion that
> builds the statistic from scratch in Python — open a lesson's card on the course map
> (`rendered/index.html`) to find its **"Another dataset →"** and **"The code behind it →"**
> links. That's **57 notebooks** in all (21 core + 18 data + 18 code). *(Next: the course
> goes online — read + run it in your browser.)*

## A note on honesty

This course makes claims from data, so it plays fair: every dataset's source and
license is logged (`raw/SOURCES.md`), and each lesson keeps "the math says,"
"the data shows," and "what this means in the world" clearly separate. It's for
study and understanding — not professional (legal, medical, financial) advice.
