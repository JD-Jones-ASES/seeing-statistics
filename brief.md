# Project brief — Statistics, by seeing it happen

**Goal (the user's own words):**
> "I'm rusty on statistics, and would like a resource that will explain things. The technical math is needed, but also, 'What does it mean?' 'What does this number do?' 'What does this result say/mean?' Can you find some real data sets from reliable sources to use as example for simulations and to build graphs, etc? For example, what's a 95% confidence interval? Teach the normal stuff, but also: pick a data set, sample from it, and make 100 confidence intervals. See how many contain the real mean, which we know. Display this in various ways. So, I want a course through a couple of semesters of university probability & statistics, starting ground floor, but learning by seeing the data and playing with it whenever possible. I don't care about curriculum alignment or anything."

**Plain-English translation:**
A self-paced, two-semester probability & statistics course built from the ground up, delivered as a folder of interactive Python notebooks. Every concept is taught on three levels — the math, what the number *means*, and what a result *says about the world* — and, wherever possible, *demonstrated by simulation on real data* rather than asserted. The signature move: treat a real dataset as the known "truth," then sample from it over and over to watch ideas like the Central Limit Theorem and the 95% confidence interval actually happen.

**Project type:**
Tier 2 · local. Notebooks run on your own computer with free, standard tools. No website, no server, no account, no cloud. Datasets are downloaded once from reliable public sources and kept read-only.

**Tools likely needed:**
- **Python** (already installed) — runs every calculation and simulation; you read and tweak, the notebook does the work.
- **Jupyter (JupyterLab)** — the notebook environment where words, math, code, and charts live together on one page you can re-run.
- **pandas** — loads and handles the real datasets (tables/CSVs).
- **NumPy** — the fast number-crunching and random-sampling engine behind the simulations.
- **Matplotlib / Seaborn** — draw the histograms, boxplots, scatterplots, and the confidence-interval "caterpillar" charts.
- **SciPy / statsmodels** — the trusted formulas for distributions, t-tests, ANOVA, chi-square, and regression (Semester 2).

**Tools NOT needed:**
- No web framework, server, or database — a folder of notebooks plus CSV files is the whole thing.
- No cloud account, API key, or paid service — everything runs offline on your machine.
- No deployment — this is yours to run locally; publishing a read-only web version is a possible *later* choice, not part of this.

**Good first milestone:**
Two notebooks you can open and run: (1) **Lesson 1 — describing & visualizing a real dataset** (center, spread, shape, with live charts), and (2) the **flagship 95% confidence-interval coverage demo** — sample a real "population" dataset 100 times, build a CI from each sample, and watch how many of the 100 actually contain the true mean (shown several ways).

**Success criteria (what a good first version is):**
- I can open a notebook, read a plain-English explanation *and* the real math, and see it demonstrated on real data.
- I can change a number (sample size, confidence level, number of repeats) and re-run to watch the result change.
- The confidence-interval demo clearly shows that ~95 of 100 intervals capture the known true mean — and what it means when one misses.
- Every dataset's source and license is logged; nothing I run uploads my data or requires an account.

**Risks / open questions:**
- Time: each lesson is ~an afternoon to work through; the course is a long, paced project (months, at your pace).
- Money: $0 — all tools and datasets are free, no card, no subscription.
- Data exposure: stays on your computer. Datasets are *downloaded from* public sources; nothing of yours is uploaded.
- Reversibility: fully reversible. Libraries live in a private toolbox (`.venv`); deleting that folder undoes the install. Notebooks and data are just files.
- Open question: Python 3.14 is very new — if any library lacks a ready-made build yet, we substitute or pin a version. (Checked during setup.)

**What would push this out of beginner scope:**
Turning this into a public, multi-user website with logins or graded quizzes that store results — that becomes a hosted web app with a server and ongoing cost. If that's ever wanted, it gets its own decision record first. A read-only static export of the notebooks (just web pages) is a safe middle option that stays local-friendly.

**Validation note (this course makes claims, so this matters):**
Datasets are logged with source + license in `raw/SOURCES.md` and kept read-only. Lessons separate three things cleanly: *the math* (what the formula computes), *what the data shows* (the result of the computation), and *real-world interpretation* (which is presented as study/illustration, never as professional advice). Simulations use the dataset as a self-contained "truth" so claims are about the data in hand, not the wider world.

**Next step:**
Finish installing the libraries, drop in the verified "spine" dataset from the dataset research, and build the two first-milestone notebooks.
