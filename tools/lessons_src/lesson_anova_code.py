r"""Lesson 16 (code variant) — "The code behind it": build one-way ANOVA from the
variance decomposition in NumPy, on the same Palmer Penguins (body mass across the
three species) as the core lesson. Same skeleton as every Phase 4 code module: the
goal -> the data in code -> build it step by step -> the idiom (groupby, vectorized
sums of squares) -> refactor into a reusable function -> now you code -> what you
learned. See MANIFEST in lessonkit.py."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # tools/
from lessonkit import md, code, save_lesson, header_md, nav_md

ID = "anova-code"

INTRO = r"""
**In Lesson 16 you called `stats.f_oneway(*groups)`** and out came an $F$ of about 343 and a vanishingly
small p-value. It worked like a magic word. This companion **opens the hood**: you'll build that $F$
*yourself*, straight from the **variance decomposition** — so an F-statistic stops being a number scipy
hands you and becomes a few lines of NumPy you could write from memory.

Same penguins, same body-mass column, same three species, same final $F$ — but the focus shifts from
*what ANOVA means* to **how you compute it in Python**. We'll go in the order a programmer actually works:

1. **The goal** — what we're building.
2. **The data in code** — load it and split it into groups with `groupby`.
3. **Build it step by step** — total / between / within sums of squares (they add up!), mean squares, the $F$-ratio and its p-value, checked against scipy.
4. **The idiom** — `groupby` for per-group means and sizes; vectorized sums of squares (no loops over rows); the SS-identity as a self-check.
5. **Refactor into a function** — wrap it all as your own reusable `one_way_anova(groups)`.
6. **Now you code** — predict-then-run exercises.
7. **What you learned** — the coding *and* the statistics.

You don't need Lesson 16 fresh in mind, but it helps. Run each grey cell with **Shift + Enter**, top to bottom.
"""


def cells():
    return [
        md(header_md(ID, intro=INTRO)),
        md(r"""
## 1) The goal

By the end of this notebook you'll have written a function that takes a list of groups (one array of
numbers per species) and returns the **F-statistic**, its two degrees of freedom, and a p-value — the same
thing `scipy.stats.f_oneway` hands you for free. Nothing here needs new statistics; everything is the
Python *underneath* that one call.

The whole test rests on a single, beautiful identity — the **variance decomposition**. The total spread of
the data splits cleanly into two pieces:

$$\underbrace{SS_{\text{total}}}_{\text{everything wobbles from the grand mean}}
  \;=\;
  \underbrace{SS_{\text{between}}}_{\text{group means wobble from the grand mean}}
  \;+\;
  \underbrace{SS_{\text{within}}}_{\text{points wobble from their own group mean}}.$$

We'll compute all three from their formulas, *prove* they add up, then turn them into the $F$-ratio.

The tools we'll use, and what each is for:

| Tool | What it is | What we use it for |
|---|---|---|
| **pandas** | tables (`DataFrame`) and columns (`Series`) | loading the CSV, splitting into groups with `groupby` |
| **NumPy** | fast math on whole arrays at once | the sums of squares (`.sum()`, `** 2`, ...) |
| **scipy.stats** | ready-made statistical tests | the answer key: `f_oneway`, and `f.sf` for the p-value |

A **group** here is just the body masses of one species, pulled out as a NumPy array. Almost everything
below is "do some arithmetic to a whole array at once."
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
from scipy import stats        # f_oneway (the answer key), f.sf (the p-value)
sl.use_course_style()

PENGUINS_URL = 'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv'
penguins = sl.load_csv('penguins.csv', url=PENGUINS_URL)
print(f'Loaded a {type(penguins).__name__} with shape {penguins.shape}  (rows, columns)')
"""),
        md(r"""
## 2) The data in code

Same cleaning as the core lesson, so our group sizes match it exactly. We keep only the rows that have
both columns we need — **`body_mass_g`** (the response, in grams) and **`species`** (the grouping) — by
dropping `NaN`s. This makes a **new, smaller table**; it never edits the file on disk (`raw/` stays
read-only).

Then we **split** the data into one array per species. The clean way to get *per-group* numbers is
pandas' `groupby`, which is the split-apply-combine pattern: split the rows by species, apply a summary,
combine into a table.
"""),
        code(r"""
penguins = penguins.dropna(subset=['body_mass_g', 'species'])
species_order = ['Adelie', 'Chinstrap', 'Gentoo']

print(f'{len(penguins)} penguins across {penguins["species"].nunique()} species\n')

# groupby = split-apply-combine: a count, mean and SD per species, in one expression.
print(penguins.groupby('species')['body_mass_g'].agg(['count', 'mean', 'std']).round(1))

# Pull each species' body masses out as its own NumPy array — this is our list of "groups".
groups = [penguins.loc[penguins['species'] == s, 'body_mass_g'].to_numpy() for s in species_order]
print('\ngroup sizes (n per species):', [len(g) for g in groups])
print('first 5 Adelie masses:', groups[0][:5])
"""),
        md(r"""
The three sample means are clearly not identical — Gentoo look much heavier (~5076 g) than Adelie
(~3700 g) and Chinstrap (~3733 g). ANOVA's job is to decide whether that gap is **bigger than the noise
inside each species**. To do that, it needs three sums of squares. Let's build them.

## 3) Build it step by step

### The grand mean and the group means

Everything keys off two kinds of average:

- the **grand mean** $\bar{x}$ — the mean of *all* the masses, ignoring species;
- each **group mean** $\bar{x}_j$ — the mean within one species, with size $n_j$.

`groupby` gives us the per-group means and sizes in one shot; the grand mean is just `.mean()` on the
whole column.
"""),
        code(r"""
all_mass = penguins['body_mass_g'].to_numpy()
grand_mean = all_mass.mean()
N = len(all_mass)        # total number of penguins
k = len(groups)          # number of groups (species)

group_means = penguins.groupby('species')['body_mass_g'].mean()   # one mean per species
group_sizes = penguins.groupby('species')['body_mass_g'].size()   # one n per species

print(f'grand mean x-bar = {grand_mean:.4f} g   over N = {N} penguins in k = {k} groups\n')
print('per-group mean and size (the n_j and x-bar_j the formulas need):')
for s in species_order:
    print(f'  {s:<10} n = {group_sizes[s]:>3}   mean = {group_means[s]:.1f} g')
"""),
        md(r"""
### The three sums of squares

Now the heart of ANOVA. A "sum of squares" is just **add up squared distances** — no division, no `ddof`
anywhere (these are raw sums; the dividing-by-degrees-of-freedom happens *later*, when we form the mean
squares). Watch the vectorization: `all_mass - grand_mean` subtracts the grand mean from **every** value
at once; `** 2` squares each; `.sum()` adds them. No loop over penguins.

$$SS_{\text{total}} = \sum_i (x_i - \bar{x})^2, \qquad
  SS_{\text{between}} = \sum_{j} n_j(\bar{x}_j - \bar{x})^2, \qquad
  SS_{\text{within}} = \sum_{j} \sum_{i \in j} (x_{ij} - \bar{x}_j)^2.$$
"""),
        code(r"""
# SST: every penguin's distance from the GRAND mean, squared and summed.
ss_total = ((all_mass - grand_mean) ** 2).sum()

# SSB: each GROUP MEAN's distance from the grand mean, weighted by group size n_j.
ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)

# SSW: each penguin's distance from its OWN group's mean, squared and summed.
ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)

print(f'SS_total   = {ss_total:16,.2f}')
print(f'SS_between = {ss_between:16,.2f}')
print(f'SS_within  = {ss_within:16,.2f}')
"""),
        md(r"""
### The identity that makes ANOVA work

Here is the payoff. Those two pieces — between and within — must **add up to the total**. That is not a
coincidence; it is an algebraic identity, and it is the very reason we can split "variance" into a signal
part and a noise part. Let's check it numerically.
"""),
        code(r"""
reconstructed = ss_between + ss_within
print(f'SS_between + SS_within = {reconstructed:16,.2f}')
print(f'SS_total               = {ss_total:16,.2f}')
print('\nSST = SSB + SSW  match?', np.isclose(ss_total, reconstructed))
print('\n-> The total spread splits perfectly into BETWEEN-group + WITHIN-group. That clean')
print('   split is what lets ANOVA compare "signal" (between) against "noise" (within).')
"""),
        md(r"""
### Mean squares and the F-ratio

A raw sum of squares grows just by having more data, so we scale each by its **degrees of freedom** (df) —
a count of how many values were free to vary — to get a **mean square**. Between-group df is $k-1$;
within-group df is $N-k$. The F-statistic is their ratio.

$$MS_{\text{between}} = \frac{SS_{\text{between}}}{k-1}, \quad
  MS_{\text{within}} = \frac{SS_{\text{within}}}{N-k}, \quad
  F = \frac{MS_{\text{between}}}{MS_{\text{within}}}.$$

The p-value comes from the **F-distribution** with those two df — `stats.f.sf(F, df_between, df_within)`
is the area to the *right* of our $F$ (the "survival function", $1 - \text{CDF}$): *how often would chance
alone produce an $F$ this big or bigger if every species truly had the same mean?*
"""),
        code(r"""
df_between = k - 1            # 3 groups -> 2
df_within  = N - k            # 342 penguins - 3 groups -> 339

ms_between = ss_between / df_between
ms_within  = ss_within / df_within
F_hand = ms_between / ms_within
p_hand = stats.f.sf(F_hand, df_between, df_within)   # area to the right of F

print(f'df_between = {df_between}   df_within = {df_within}\n')
print(f'MS_between = {ms_between:14,.2f}')
print(f'MS_within  = {ms_within:14,.2f}')
print(f'\nF = MS_between / MS_within = {F_hand:.4f}')
print(f'p = f.sf(F, {df_between}, {df_within})    = {p_hand:.4e}')
"""),
        md(r"""
### Check it against the answer key

We built $F$ from raw sums of squares; scipy does the identical bookkeeping internally. They should agree
to the last decimal — and the p-value too.
"""),
        code(r"""
F_scipy, p_scipy = stats.f_oneway(*groups)

print(f'our by-hand  :  F = {F_hand:.4f},   p = {p_hand:.4e}')
print(f'scipy f_oneway:  F = {F_scipy:.4f},   p = {p_scipy:.4e}')
print('\nmatch?', np.isclose(F_hand, F_scipy), np.isclose(p_hand, p_scipy))
print(f'\n-> F = {F_hand:.0f} is enormous, p = {p_hand:.1e} is astronomically small: the spread BETWEEN')
print('   the species means dwarfs the wobble WITHIN species. At least one species really differs.')
"""),
        md(r"""
### Why not just run all the pairwise t-tests? (the multiple-comparisons trap)

With three groups it's tempting to skip ANOVA and t-test every pair instead (Adelie vs Chinstrap, Adelie
vs Gentoo, Chinstrap vs Gentoo). The trouble: each t-test at $\alpha = 0.05$ has its own 5% chance of a
false alarm, and those chances **pile up** across the family of tests. Let's *see* it — build a world
where **nothing differs** (every group drawn from the same distribution), run all the pairwise tests, and
count how often *at least one* fires by accident.
"""),
        code(r"""
rng = np.random.default_rng(0)   # seeded so the rates below are reproducible

def family_has_false_alarm(k_groups, n_per_group=30, alpha=0.05):
    '''Draw k groups from the SAME distribution (no real difference),
       run every pairwise t-test, return True if ANY came out "significant".'''
    gs = [rng.normal(0, 1, size=n_per_group) for _ in range(k_groups)]
    for i in range(k_groups):
        for j in range(i + 1, k_groups):
            if stats.ttest_ind(gs[i], gs[j]).pvalue < alpha:
                return True          # a FALSE positive — there was no real difference
    return False

TRIALS = 4000
for kg in [2, 3, 5, 10]:
    n_pairs = kg * (kg - 1) // 2
    fwer = np.mean([family_has_false_alarm(kg) for _ in range(TRIALS)])
    print(f'{kg:>2} groups ({n_pairs:>2} pairwise tests):  '
          f'at-least-one-false-alarm rate = {fwer*100:4.1f}%   (each single test is only 5%)')
"""),
        md(r"""
With **2** groups (1 test) the false-alarm rate is the promised ~5%. But by **5** groups it's already
~29%, and by **10** groups ~61% — even though, by construction, **nothing is different**. That is the
multiple-comparisons trap, and it's exactly why we run **one** omnibus ANOVA at a controlled 5% instead of
spraying t-tests across every pair. (The core lesson goes deeper on this, and on the post-hoc Tukey HSD
that finds *which* pairs differ while staying honest.)

## 4) The idiom — `groupby` and vectorized sums of squares

Three habits made the build above short and fast, and they're the patterns to reuse:

- **`groupby` for per-group numbers.** `penguins.groupby('species')['body_mass_g'].agg([...])` gives the
  count, mean and SD of every group in one expression — no manual looping to bin the rows.
- **Vectorized sums of squares.** `((g - g.mean()) ** 2).sum()` computes a whole group's within-sum in one
  line: subtract the mean from every element at once, square, add. No `for` loop over individual penguins.
- **The SS identity as a correctness check.** `np.isclose(SST, SSB + SSW)` is a cheap assertion that your
  decomposition is right — if it ever fails, you've made an arithmetic slip.

One subtlety worth stating plainly: the sums of squares use **no `ddof`** — they are raw sums of squared
deviations. The "divide by something" only happens when we form the mean squares, and there we divide by
the **degrees of freedom** ($k-1$ and $N-k$), *not* by $n$ or $n-1$. Get that right and your $F$ matches
`f_oneway` exactly, as we just saw.
"""),
        code(r"""
# The whole F-ratio, expressed compactly with the idioms above:
gm = all_mass.mean()
ssb = sum(g.size * (g.mean() - gm) ** 2 for g in groups)          # between
ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)            # within
F_compact = (ssb / (k - 1)) / (ssw / (N - k))

print(f'compact F = {F_compact:.4f}')
print('matches scipy?', np.isclose(F_compact, stats.f_oneway(*groups).statistic))
print('\nNo loop over rows anywhere — every sum of squares is a vectorized array operation,')
print('and groupby (above) did the splitting. That is the idiom.')
"""),
        md(r"""
## 5) Refactor into a reusable function

You've now written the same handful of lines — grand mean, between-sum, within-sum, mean squares, ratio,
p-value — a few times. The programmer's reflex is to **wrap repeated work in a function**: name it, give
it inputs, `return` the result. Then "run a one-way ANOVA" is one call you can reuse on *any* list of
groups — flipper lengths, a different dataset entirely.
"""),
        code(r"""
def one_way_anova(groups):
    '''One-way ANOVA from the variance decomposition.

    groups: a list of 1-D arrays, one per group.
    Returns dict(F, df_between, df_within, p) — same F and p as scipy.stats.f_oneway.
    '''
    groups = [np.asarray(g, dtype=float) for g in groups]
    y = np.concatenate(groups)
    grand = y.mean()
    N = y.size
    k = len(groups)
    ss_between = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ss_within  = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b, df_w = k - 1, N - k
    F = (ss_between / df_b) / (ss_within / df_w)
    return {'F': F, 'df_between': df_b, 'df_within': df_w, 'p': stats.f.sf(F, df_b, df_w)}

result = one_way_anova(groups)
print('one_way_anova(groups) ->')
for key, val in result.items():
    print(f'  {key:<11} = {val}')

# Re-check the function against scipy, the answer key:
F_scipy, p_scipy = stats.f_oneway(*groups)
print(f'\nscipy f_oneway:  F = {F_scipy:.4f},  p = {p_scipy:.4e}')
print('match?', np.isclose(result['F'], F_scipy), np.isclose(result['p'], p_scipy))
"""),
        md(r"""
A function is a contract: *give me a list of groups, I'll give you $F$, both df's, and a p-value.* Once
it's correct, you stop thinking about sums of squares and start thinking in **groups and F-ratios**. Let's
prove the reuse by pointing it at a *different* response column — flipper length — with no copy-paste of
formulas.
"""),
        code(r"""
fl = penguins.dropna(subset=['flipper_length_mm'])
flipper_groups = [fl.loc[fl['species'] == s, 'flipper_length_mm'].to_numpy() for s in species_order]

flip = one_way_anova(flipper_groups)
print('ANOVA on flipper_length_mm across species:')
print(f"  F = {flip['F']:.4f}   df = ({flip['df_between']}, {flip['df_within']})   p = {flip['p']:.4e}")
print('matches scipy?', np.isclose(flip['F'], stats.f_oneway(*flipper_groups).statistic))
print('\nSame function, brand-new column — flipper length separates the species even more')
print(f"strongly than body mass did (F = {flip['F']:.0f} vs {result['F']:.0f}).")
"""),
        md(r"""
## 6) Now you code

Write the code yourself. Add a new cell (the **+** button in the toolbar) and try each — predict the
answer before you run it.

1. **Recover the means from the function's pieces.** Inside a fresh cell, recompute `ss_between` for the
   body-mass `groups` *only from* the group means and sizes (`sum(n_j * (mean_j - grand)**2 ...)`), then
   confirm it equals the `ss_between` from Section 3 with `np.isclose`. (Point: between-group spread needs
   *only* the group summaries, never the raw points.)

2. **Mean squares by hand, then the F-ratio.** Take `flipper_groups`, compute `ss_within` and
   `ss_within / (N_fl - k)` (the within mean square), and likewise `ms_between`. Divide them and check the
   ratio equals `one_way_anova(flipper_groups)['F']`.

3. **Two groups: ANOVA *is* the t-test.** Pull out just Adelie and Chinstrap body masses as a 2-element
   list and call `one_way_anova([...])`. Separately run `stats.ttest_ind(adelie, chinstrap)`. Confirm the
   F equals the t **squared** (`t**2`) and the p-values match — with two groups, ANOVA *is* the
   two-sample t-test.

4. **Break the identity on purpose.** Take the body-mass `groups`, but compute a *wrong* `ss_within` using
   the **grand** mean instead of each group's own mean (`((g - grand_mean)**2).sum()`). Add it to
   `ss_between` and check whether it still equals `ss_total`. (It won't — the decomposition only closes
   when "within" is measured from each group's *own* mean. That's the whole trick.)

## What you learned

**Coding**
- **`groupby(...).agg([...])`** turns "a number per group" (count, mean, SD, size) into one line — the
  split-apply-combine idiom that feeds every per-group quantity ANOVA needs.
- **Vectorized sums of squares**: `((g - g.mean()) ** 2).sum()` computes a whole group's spread with no
  `for` loop over rows — subtract, square, add, all at once.
- **Use an identity as a self-check**: `np.isclose(SST, SSB + SSW)` is a cheap assertion that your
  decomposition is correct.
- **Refactor repeated work into a function** with clear inputs and a `return` (here a dict of
  `F, df_between, df_within, p`), then reuse it on any list of groups — body mass, flipper length, anything.
- **Check against an answer key**: every number was confirmed against `scipy.stats.f_oneway` to the decimal.

**Statistics (now demystified)**
- ANOVA rests on the **variance decomposition** $SS_{\text{total}} = SS_{\text{between}} + SS_{\text{within}}$
  — you proved the two pieces add to the whole.
- Sums of squares are **raw** (no `ddof`); the division is by **degrees of freedom** — $k-1$ for between
  and $N-k$ for within — when you form the mean squares. That's what makes $F$ match `f_oneway` exactly.
- $F = MS_{\text{between}} / MS_{\text{within}}$ is large when group means are spread far apart relative to
  the noise inside groups; here $F \approx 344$ with $p \approx 3\times10^{-82}$.
- The **p-value** is the right-tail area of the **F-distribution** with $(k-1,\,N-k)$ df — `stats.f.sf(F, df_between, df_within)`.
- **Running every pairwise t-test inflates the false-positive rate** (you simulated ~29% at 5 groups, ~61%
  at 10 — with nothing truly different), which is exactly why one omnibus ANOVA is the honest move.

**Back to the core lesson** ([Lesson 16: Comparing many groups: ANOVA](16-anova.ipynb)) for the *meaning*
of these numbers — the F-distribution by simulation, Tukey HSD for *which* pairs differ, and ANOVA's
assumptions.
"""),
        md(nav_md(ID)),
    ]


if __name__ == "__main__":
    save_lesson(ID, cells())
