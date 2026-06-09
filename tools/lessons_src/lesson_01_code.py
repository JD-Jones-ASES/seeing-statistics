r"""Lesson 1 (code variant) — "The code behind it": build the describing-data
summaries from scratch in Python (NumPy/pandas), on the same Palmer Penguins as
the core lesson. This is the FIRST code-variant module and the template the rest
of Phase 4 follows: a distinct skeleton — the goal -> the data in code -> build
it step by step -> the idiom (vectorized vs loop, tidy pandas) -> refactor into a
reusable function -> now you code -> what you learned. See MANIFEST in lessonkit.py."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # tools/
from lessonkit import md, code, save_lesson, header_md, nav_md

ID = "01-code"

INTRO = r"""
**In Lesson 1 you described penguins with pandas one-liners** — `mass.mean()`, `mass.std()`,
`flipper.corr(mass)`. They worked like magic words. This companion **opens the hood**: you'll build
those same summaries *yourself*, so a "mean" stops being a method you call and becomes three lines of
code you could write from memory.

Same penguins, same final numbers — but the focus shifts from *what the statistic means* to **how you
compute it in Python**. We'll go in the order a programmer actually works:

1. **The goal** — what we're building.
2. **The data in code** — load it, look at its shape, select columns and rows.
3. **Build it step by step** — turn the math formulas for the mean, variance, SD, z-score and correlation into NumPy.
4. **The idiom** — the *vectorized* way vs. a slow Python loop, and split-apply-combine with `groupby`.
5. **Refactor into a function** — wrap it all up as your own reusable `describe()`.
6. **Now you code** — write-it-yourself exercises.
7. **What you learned** — the coding *and* the statistics.

You don't need Lesson 1 fresh in mind, but it helps. Run each grey cell with **Shift + Enter**, top to bottom.
"""


def cells():
    return [
        md(header_md(ID, intro=INTRO)),
        md(r"""
## 1) The goal

By the end of this notebook you'll have written a function that takes a column of numbers and returns its
center, spread and shape — the same summary `pandas` hands you for free — plus a from-scratch correlation
and a per-group summary table. Nothing here needs new statistics; everything is the Python *underneath*
the one-liners you already ran.

The tools we'll use, and what each is for:

| Tool | What it is | What we use it for |
|---|---|---|
| **pandas** | tables (`DataFrame`) and columns (`Series`) | loading the CSV, selecting, grouping |
| **NumPy** | fast math on whole arrays at once | the actual formulas (sum, sqrt, ...) |
| **matplotlib** | plotting | a histogram we draw ourselves |

A **`DataFrame`** is a spreadsheet in code: named columns, numbered rows. A single column is a **`Series`** —
an array of values with an index. Almost everything below is "do some arithmetic to a whole Series at once."
"""),
        code(r"""
# --- Setup: load our helpers and the data --------------------------------
import sys, pathlib
for _p in [pathlib.Path.cwd(), pathlib.Path.cwd().parent]:
    if (_p / 'lib' / 'statslab.py').exists():
        sys.path.insert(0, str(_p / 'lib')); break
import statslab as sl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sl.use_course_style()

PENGUINS_URL = 'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv'
penguins = sl.load_csv('penguins.csv', url=PENGUINS_URL)
print(f'Loaded a {type(penguins).__name__} with shape {penguins.shape}  (rows, columns)')
"""),
        md(r"""
## 2) The data in code

Before any statistics, a programmer *interrogates the object*: what type is it, how big is it, what are the
columns, what do the first rows look like? Four habits worth keeping:

- `.shape` → `(rows, columns)` as a tuple.
- `.columns` → the column names.
- `.dtypes` → the **type** of each column (`float64` = decimal number, `object` = text).
- `.head()` → the first few rows, to eyeball it.
"""),
        code(r"""
print('shape  :', penguins.shape)
print('columns:', list(penguins.columns))
print('\ndtypes (the type stored in each column):')
print(penguins.dtypes)
penguins.head()
"""),
        md(r"""
### Selecting columns and rows

- **One column** → `penguins['body_mass_g']` gives a `Series`.
- **A boolean mask** → `penguins['species'] == 'Gentoo'` makes a column of `True`/`False`; putting it
  *inside the brackets* keeps only the `True` rows. This is the single most useful pattern in pandas.

### Missing values

Real data has holes. A few penguins are missing a measurement, stored as `NaN` ("not a number"). Math on a
`NaN` silently spreads it, so we drop the rows missing the columns we need with `.dropna(subset=[...])`.
This makes a **new, smaller table** — it never edits the file on disk (our `raw/` data stays read-only).
"""),
        code(r"""
print('rows with ANY missing value:', penguins.isna().any(axis=1).sum())

penguins = penguins.dropna(subset=['body_mass_g', 'flipper_length_mm', 'sex'])
print('rows kept after dropna       :', len(penguins))

# Pull out the column we'll study most. It is a Series of 333 numbers.
mass = penguins['body_mass_g']
print('\nmass is a', type(mass).__name__, 'of', len(mass), 'values; first 5:')
print(mass.head().to_list())

# A boolean mask in action: just the Gentoos.
gentoo_mask = penguins['species'] == 'Gentoo'
print(f'\nGentoo penguins: {gentoo_mask.sum()} of {len(penguins)}')
"""),
        md(r"""
## 3) Build it step by step

Now the heart of it. Each statistic is a one-line `pandas` method *and* a short formula — we'll write the
formula in NumPy and check the two agree to the last decimal. Once you've built it, the method is no longer
a mystery box.

### The mean

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i \qquad\text{— add them all up, divide by how many.}$$

In code that is literally `mass.sum() / len(mass)`. `numpy` adds a whole array for you, so no loop needed.
"""),
        code(r"""
n = len(mass)
mean_by_hand = mass.sum() / n          # the formula, verbatim
mean_pandas  = mass.mean()             # the one-liner from Lesson 1

print(f'mean by hand  : {mean_by_hand:.6f}')
print(f'mass.mean()   : {mean_pandas:.6f}')
print('match?', np.isclose(mean_by_hand, mean_pandas))
"""),
        md(r"""
### Variance and standard deviation

Variance is the **average squared distance from the mean** — but divided by $n-1$, not $n$, for a sample
(that "minus one" is **Bessel's correction**; it keeps the estimate from running too small):

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2, \qquad s=\sqrt{s^2}.$$

Watch the vectorization: `mass - mean_by_hand` subtracts the mean from **every** value at once, giving a
whole array of deviations. `** 2` squares each. `.sum()` adds them. No loop anywhere.
"""),
        code(r"""
deviations = mass - mean_by_hand        # an array of 333 differences, all at once
sq_dev     = deviations ** 2            # square each one
var_by_hand = sq_dev.sum() / (n - 1)    # divide by n-1  (Bessel's correction)
sd_by_hand  = np.sqrt(var_by_hand)

print(f'variance by hand : {var_by_hand:,.4f}')
print(f'mass.var(ddof=1) : {mass.var(ddof=1):,.4f}')
print(f'SD by hand       : {sd_by_hand:.4f}')
print(f'mass.std(ddof=1) : {mass.std(ddof=1):.4f}')
print('\nmatch?', np.isclose(var_by_hand, mass.var(ddof=1)), np.isclose(sd_by_hand, mass.std(ddof=1)))
print("(ddof=1 is how pandas/numpy say 'divide by n-1'. The default in numpy's np.std is ddof=0 — divide")
print(" by n — so always pass ddof=1 for a sample, or your SD comes out a hair too small.)")
"""),
        md(r"""
### z-scores — one formula, applied to the whole column

A **z-score** says "how many SDs from the mean": $z = (x - \bar{x}) / s$. Because subtraction and division
work element-by-element, the entire column of z-scores is *one expression* — no loop over penguins.
"""),
        code(r"""
z = (mass - mean_by_hand) / sd_by_hand     # 333 z-scores in one line
print('first 5 z-scores:', np.round(z.head().to_numpy(), 2))
print(f'mean of z (should be ~0): {z.mean():.2e}')
print(f'SD of z   (should be ~1): {z.std(ddof=1):.4f}')

# Boolean mask + .mean() is the idiom for "what fraction satisfies a condition?"
within1 = (z.abs() < 1).mean() * 100
print(f'\n{within1:.0f}% of penguins are within 1 SD of the mean — (z.abs() < 1) is a True/False column,')
print('and the mean of True/False is just the proportion that are True.')
"""),
        md(r"""
### Correlation, from its formula

Lesson 1's `flipper.corr(mass)` is **Pearson's $r$**. Its formula looks scary but is just "do the
deviations of $x$ and $y$ move together?", scaled to land between $-1$ and $+1$:

$$r = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum (x_i-\bar{x})^2}\;\sqrt{\sum (y_i-\bar{y})^2}}.$$

Every $\sum$ is a `.sum()` over a vectorized product. Build it and check against the one-liner.
"""),
        code(r"""
x = penguins['flipper_length_mm']
y = penguins['body_mass_g']
dx = x - x.mean()
dy = y - y.mean()

r_by_hand = (dx * dy).sum() / np.sqrt((dx**2).sum() * (dy**2).sum())
r_pandas  = x.corr(y)

print(f'r by hand   : {r_by_hand:.6f}')
print(f'x.corr(y)   : {r_pandas:.6f}')
print('match?', np.isclose(r_by_hand, r_pandas))
print(f'\n-> r = {r_by_hand:.2f}: longer flippers go strongly with heavier penguins.')
"""),
        md(r"""
## 4) The idiom — vectorized vs. a loop, and `groupby`

### Why we never wrote a loop

You *could* compute the mean with a Python `for` loop. It gives the same answer but is slower and noisier to
read. Compare the two — the vectorized one-liner is the idiom you want.
"""),
        code(r"""
# The loop way (how you might first think of it):
total = 0.0
for value in mass:
    total += value
mean_loop = total / len(mass)

# The vectorized way (what NumPy/pandas is for):
mean_vec = mass.mean()

print(f'loop result      : {mean_loop:.4f}')
print(f'vectorized result: {mean_vec:.4f}   <- same number, one line, and far faster on big data')

# Let's actually time them on a million numbers so the difference is visible.
import time
big = np.random.default_rng(0).normal(size=1_000_000)

t0 = time.perf_counter()
s = 0.0
for v in big:
    s += v
loop_mean = s / len(big)
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
vec_mean = big.mean()
t_vec = time.perf_counter() - t0

print(f'\nMean of 1,000,000 numbers:')
print(f'  Python loop : {t_loop*1000:7.1f} ms')
print(f'  NumPy .mean(): {t_vec*1000:7.1f} ms')
print(f'  NumPy is roughly {t_loop/max(t_vec,1e-9):.0f}x faster — and it is one short line.')
"""),
        md(r"""
### Split → apply → combine with `groupby`

In Lesson 1 you compared species. The pattern behind "a number per group" is **split-apply-combine**:
*split* the rows by a category, *apply* a summary to each chunk, *combine* the answers into a table.
`groupby` does all three in one expression.
"""),
        code(r"""
# One statistic per species:
print(penguins.groupby('species')['body_mass_g'].mean().round(1))

print('\n--- several statistics at once with .agg() ---')
summary = penguins.groupby('species')['body_mass_g'].agg(['count', 'mean', 'std', 'min', 'max']).round(1)
print(summary)
print("\nThat table is the split-apply-combine pattern: split by species, apply count/mean/std/...,")
print("combine into one tidy DataFrame. The same idea powers pivot tables and SQL's GROUP BY.")
"""),
        md(r"""
## 5) Refactor into a reusable function

You've now written the same four lines (mean, var, SD, ...) a few times. The programmer's reflex is to
**wrap repeated work in a function**: name it, give it inputs, `return` the result. Then "describe a column"
is one call you can reuse on *any* numeric column — flipper length, bill depth, a different dataset entirely.
"""),
        code(r"""
def describe_column(series):
    '''Return center, spread and shape for a numeric Series — our own .describe().'''
    s = series.dropna()
    n = len(s)
    mean = s.sum() / n
    var = ((s - mean) ** 2).sum() / (n - 1)
    sd = np.sqrt(var)
    q1, q3 = s.quantile([0.25, 0.75])
    return {
        'n': n,
        'mean': round(mean, 2),
        'sd': round(sd, 2),
        'min': s.min(),
        'median': s.median(),
        'max': s.max(),
        'iqr': round(q3 - q1, 2),
    }

# Reuse it on three different columns — no copy-paste of formulas:
for col in ['body_mass_g', 'flipper_length_mm', 'bill_length_mm']:
    print(f'{col:20}', describe_column(penguins[col]))
"""),
        md(r"""
A function is a contract: *give me a Series, I'll give you a dict of summaries.* Once it's correct, you stop
thinking about the formulas and start thinking in **columns and summaries** — exactly the leap from "writing
statistics" to "doing statistics in code." (And yes, `series.describe()` exists; the point was to *build* it
so you know what it's doing.)
"""),
        md(r"""
## 6) Now you code

Write the code yourself. Add a new cell (the **+** button in the toolbar) and try each — predict the answer
before you run it.

1. **A function for the IQR.** Write `iqr(series)` that returns `Q3 − Q1`. Use `series.quantile([0.25, 0.75])`,
   which gives the two values back; subtract them. Test it on `penguins['body_mass_g']` and check it matches
   the `iqr` field from `describe_column`.

2. **Correlation as a function.** Wrap the Section-3 correlation code into `pearson(x, y)` that returns $r$
   for two Series. Test `pearson(penguins['flipper_length_mm'], penguins['body_mass_g'])` against
   `x.corr(y)` — they should agree to many decimals.

3. **Describe by group, your way.** Without using `.agg(...)`, loop over the three species, slice each with a
   boolean mask (`penguins[penguins['species'] == s]`), and `print(s, describe_column(...))`. Compare your
   output to the `groupby` table from Section 4.

4. **Standardize a whole table.** Compute z-scores for *every* numeric measurement at once:
   `num = penguins[['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']]` then
   `(num - num.mean()) / num.std(ddof=1)`. Confirm each column's new mean is ~0 and SD ~1 with `.mean()` and
   `.std()`. (pandas lines up the columns by name for you — that's broadcasting across a DataFrame.)

## What you learned

**Coding**
- A `DataFrame` is a table; a column is a `Series`. Interrogate any new data with `.shape`, `.columns`,
  `.dtypes`, `.head()`.
- **Boolean masks** (`df[df['col'] == value]`) are how you filter rows; the **mean of a True/False column is a
  proportion**.
- **Vectorization** — arithmetic on a whole array at once (`mass - mean`, `dev ** 2`, `.sum()`) — replaces
  loops, reads cleaner, and runs orders of magnitude faster.
- **split-apply-combine** via `groupby(...).agg([...])` turns "a number per group" into one line.
- **Refactor repeated work into a function** with clear inputs and a `return`, then reuse it everywhere.

**Statistics (now demystified)**
- `mean`, `var`/`std` (with the $n-1$ **Bessel** divisor — `ddof=1`), `z`-scores, and Pearson's `r` are each
  just a short formula; you reproduced every pandas number to the decimal.
- `numpy`'s default `ddof=0` divides by $n$; pass `ddof=1` for a **sample** SD, or it comes out slightly small.

**Back to the core lesson** ([Lesson 1: Describing a real dataset](01-describing-data.ipynb)) for the
*meaning* of these numbers, or read on through the course.
"""),
        md(nav_md(ID)),
    ]


if __name__ == "__main__":
    save_lesson(ID, cells())
