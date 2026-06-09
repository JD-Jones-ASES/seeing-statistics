r"""Lesson 00 — How these notebooks work (orientation for Jupyter/Python newcomers)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # tools/
from lessonkit import md, code, save_lesson, header_md, nav_md

NUM = 0

INTRO = r"""
**Brand new to this? You're in the right place.** This page has **no statistics** — it's a five-minute tour of *how to drive a notebook* so the real lessons feel easy. You do **not** need to know how to code. You'll mostly **read**, **press one key to run a cell**, and **look at what appears**. Let's go.
"""


def cells():
    return [
        md(header_md(NUM, intro=INTRO)),
        md(r"""
## 1) What is a "notebook"?

A **notebook** is a single scrollable page made of stacked blocks called **cells**, read top to bottom — like a document where some paragraphs can *run*. There are just **two kinds of cell**:

- **Text cells** (like this one) — explanations, headings, and the math. You just read them.
- **Code cells** — the **grey boxes**. Each contains a little Python. When you *run* one, the computer does the work and shows the result right underneath.

You got here by double-clicking **`start-jupyter.bat`** (a `.bat` file is just a launcher you double-click). That opened **JupyterLab** — the free app these lessons run inside — in your web browser, along with a small **black window** that quietly runs everything behind the scenes. Leave that black window open; closing it stops the course. On the left is a file list of all the lessons; the big area is the lesson you're reading now.
"""),
        md(r"""
## 2) How to run a cell — the one move you need

1. **Click** a grey code cell once (a blue bar appears on its left).
2. Press **Shift + Enter** (hold Shift, tap Enter).
3. The cell runs, its result appears below it, and the cursor hops to the next cell.

On the left of a code cell you'll see `[ ]`. While it runs it shows `[*]` (busy), then a **number** like `[1]` when it's done — that number just counts the cells you've run, in order.

**Try it now** on the grey cell just below: click it, then press **Shift + Enter**.
"""),
        code(r"""
# This is a code cell. The lines starting with '#' are notes to you (Python ignores them).
import numpy as np
import matplotlib.pyplot as plt

print("✅ You just ran your first cell! See the number that appeared in [ ] on the left.")

# And here's a tiny chart, so you can watch a picture appear:
x = np.arange(1, 11)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(x, x**2, marker='o', color='#4c72b0')
ax.set_title("A practice chart — you made this appear by running the cell")
ax.set_xlabel("x"); ax.set_ylabel("x × x")
plt.show()
"""),
        md(r"""
## 3) Reading the output

Whatever a code cell produces shows up **directly beneath it**:

- **Printed words or numbers** (from `print(...)`) — like the green check mark above.
- **Tables** — neat grids of data.
- **Charts** — histograms, scatterplots, and the like.

That output is the *point*. The code is just the recipe; the thing underneath is what it cooked.
"""),
        md(r"""
## 4) Run cells in order, top to bottom

A lesson builds on itself, so run the cells **in order from the top**. The **very first code cell** of every lesson is the *setup* cell — it loads the tools and the dataset. If a later cell complains that something "is not defined," it almost always means an earlier cell (usually the setup one) hasn't been run yet.

The easy fix for "I think I ran things out of order": in the top menu choose **Run → Run All Cells** (or **Kernel → Restart Kernel and Run All Cells…** — the *kernel* is just the engine that runs your code). That runs everything cleanly from the top.
"""),
        md(r"""
## 5) It's safe to experiment

You **cannot break anything** by running or re-running cells. Re-running a cell just recomputes it. The lessons *invite* you to change a number (a sample size, a random seed, the number of bins) and re-run to see what happens — that's the whole idea of "seeing it happen."

If the notebook ever gets into a weird state, **Kernel → Restart Kernel and Run All Cells…** is the universal reset button. And nothing you do here touches the original data files (more on that below).
"""),
        md(r"""
## 6) Why do the numbers usually stay the same when I re-run?

Two reasons, and both are on purpose:

- **Fixed facts don't change.** Many results come from a whole dataset (an average, a count). Re-running just recomputes the same answer.
- **A fixed "random seed."** The simulations use a line like `rng = np.random.default_rng(12)`. That seed makes the "random" draws come out the *same* every time, so your results match the lesson's. **Want to see the randomness wiggle?** Change `12` to any other number and re-run — you'll get a different (but similar) result. That's not a bug; it's the lesson.
"""),
        md(r"""
## 7) Where does the data come from? (and is it safe?)

The real datasets live in the course's **`raw/`** folder. The first time a lesson needs one, it **downloads a single copy** from a public source and saves it there; after that it loads from your computer, so the course works **offline**.

Two promises, by design:

- **Your data is read-only.** The course only ever *reads* those files — it never edits or deletes them.
- **Nothing of yours is uploaded.** Data flows *to* your machine from public sources; nothing goes out. No account, no sign-in, no cost.

Every dataset's source and licence is written down in `raw/SOURCES.md`, so any chart can be traced back to where its numbers came from.
"""),
        md(r"""
## 8) If something looks broken

- **Red text under a cell** = an error. Don't panic — it's usually because an earlier cell wasn't run. Try **Kernel → Restart Kernel and Run All Cells…**.
- **A cell seems stuck on `[*]`** = it's still working (some simulations take a few seconds). Give it a moment.
- **The black command window** that opened when you launched the course must **stay open** — it's the engine ("kernel") running your notebooks. Closing it stops the course; re-launch with `start-jupyter.bat`.
- **Just want to read, not run?** Double-click `view-lessons.bat` (or open `rendered/index.html`) for finished pages with every chart already drawn — no setup at all.
"""),
        md(r"""
## You're ready

That's the entire skill: **read the text, click a grey cell, press Shift + Enter, look below.** Everything else is statistics, which the lessons teach from the ground up.

Each lesson explains every idea on **three levels** — *the math* (the formula), *the meaning* (what the number is), and *the interpretation* (what it says about the world) — and shows it happening on real data.

**Next:** Lesson 1 — describe your first real dataset (344 Antarctic penguins).
"""),
        md(nav_md(NUM)),
    ]


if __name__ == "__main__":
    save_lesson(NUM, cells())
