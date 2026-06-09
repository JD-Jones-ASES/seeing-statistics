r"""Lesson 15 (code variant) — "The code behind it": build the t-statistic from
scratch in Python — one-sample, Welch two-sample, and the paired trick — on the
same ToothGrowth + Swim data as the core lesson, each checked against
scipy.stats.ttest_*. Follows the Phase-4 code-variant skeleton: the goal -> the
data in code -> build it step by step -> the idiom -> refactor into a reusable
function -> now you code -> what you learned. See MANIFEST in lessonkit.py."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # tools/
from lessonkit import md, code, save_lesson, header_md, nav_md

ID = "t-tests-code"

INTRO = r"""
**In Lesson 15 you ran three t-tests with one-line library calls** — `stats.ttest_1samp`,
`stats.ttest_ind(..., equal_var=False)`, `stats.ttest_rel`. They each printed a `t` and a `p` and you
trusted them. This companion **opens the hood**: you'll build that `t` *yourself*, straight from the
formula, so a t-statistic stops being a number SciPy hands you and becomes three lines of arithmetic you
could write from memory.

Same guinea pigs, same swimmers, same final numbers — but the focus shifts from *what the test means* to
**how `t` is actually computed**. We'll go in the order a programmer works:

1. **The goal** — what we're building.
2. **The data in code** — load it the same way the core lesson does, and pull out the groups.
3. **Build it step by step** — the one-sample, Welch two-sample, and paired `t`, each from its formula,
   each checked against `scipy.stats.ttest_*` (print `match? True`).
4. **The idiom** — vectorized differences for the paired test, and the Welch degrees-of-freedom formula.
5. **Refactor into functions** — wrap it all as reusable `t_test_one_sample()` and `welch_t()`.
6. **Now you code** — write-it-yourself exercises.
7. **What you learned** — the coding *and* the statistics.

You don't need Lesson 15 fresh in mind, but it helps. Run each grey cell with **Shift + Enter**, top to bottom.
"""


def cells():
    return [
        md(header_md(ID, intro=INTRO)),
        md(r"""
## 1) The goal

By the end of this notebook you'll have written the t-statistic three ways — for a single group against a
target, for two independent groups (Welch's version), and for matched pairs — and confirmed each one
reproduces `scipy.stats.ttest_*` to the last decimal. Nothing here needs new statistics; everything is the
Python *underneath* the one-line calls you already ran.

The whole lesson rests on **one formula**, a signal-to-noise ratio:

$$t = \frac{\text{estimate} - \text{null value}}{\text{standard error}}.$$

The estimate is a sample mean (or a difference of means); the standard error is how much a mean of that
size typically wobbles. Every test below is just *what we plug into the numerator and denominator*.

The tools we'll use, and what each is for:

| Tool | What it is | What we use it for |
|---|---|---|
| **pandas** | tables (`DataFrame`) and columns (`Series`) | loading the CSVs, selecting groups |
| **NumPy** | fast math on whole arrays at once | the formulas (`mean`, `std`, `sqrt`, ...) |
| **scipy.stats** | a library of ready-made tests | the `ttest_*` calls we check *against* |

One rule we'll repeat like a mantra: a **sample** standard deviation divides by $n-1$, written `ddof=1` in
NumPy/pandas. Get that wrong and your `t` won't match SciPy — so it appears in every formula below.
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
from scipy import stats
sl.use_course_style()

# Same files + URLs + columns as the core Lesson 15 — so every number lines up.
TOOTH_URL = 'https://vincentarelbundock.github.io/Rdatasets/csv/datasets/ToothGrowth.csv'
SWIM_URL = 'https://www.openintro.org/data/csv/swim.csv'

tooth = sl.load_csv('ToothGrowth.csv', url=TOOTH_URL)
swim = sl.load_csv('swim.csv', url=SWIM_URL)

print(f'ToothGrowth: {type(tooth).__name__} with shape {tooth.shape}  (rows, columns)')
print(f'Swim:        {type(swim).__name__} with shape {swim.shape}')
"""),
        md(r"""
## 2) The data in code

Before any formula, a programmer *interrogates the object*: how big is it, what are the columns, what do the
first rows look like? Then we pull out exactly the arrays we'll compute on.

- **ToothGrowth** — 60 guinea pigs, each given vitamin C as orange juice (`OJ`) or ascorbic acid (`VC`);
  `len` is the tooth-cell length (the response). We'll group by `supp`.
- **Swim** — 12 swimmers, each timed in a wetsuit *and* a swimsuit; the two velocity columns are
  `wet.suit.velocity` and `swim.suit.velocity`. The same swimmer appears in both columns — that's the
  pairing.
"""),
        code(r"""
print('ToothGrowth columns:', list(tooth.columns))
print('supplement values  :', sorted(tooth['supp'].unique()))
print('Swim columns        :', list(swim.columns))
print()

# A boolean mask selects one group's response as a NumPy array (same as the core lesson):
oj = tooth.loc[tooth['supp'] == 'OJ', 'len'].to_numpy()
vc = tooth.loc[tooth['supp'] == 'VC', 'len'].to_numpy()
print(f'OJ group: {len(oj)} pigs   first 5 lengths: {oj[:5]}')
print(f'VC group: {len(vc)} pigs   first 5 lengths: {vc[:5]}')

# The two paired columns, one value per swimmer:
wet = swim['wet.suit.velocity'].to_numpy()
suit = swim['swim.suit.velocity'].to_numpy()
print(f'\nSwim: {len(wet)} swimmers   first 3 wetsuit speeds: {wet[:3]}   swimsuit: {suit[:3]}')
"""),
        md(r"""
## 3) Build it step by step

Now the heart of it. Each test is a one-line `scipy` call *and* a short formula — we'll write the formula in
NumPy and check the two agree to the last decimal. Once you've built it, the library call is no longer a
mystery box.

### One-sample t — a mean against a target

The question from the core lesson: is the **OJ** group's mean tooth growth different from a benchmark of
**18 units**? The formula is a z-score with the *sample* spread in the denominator:

$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}, \qquad df = n - 1.$$

The two-sided p-value is the area in *both* tails beyond $|t|$: `2 * stats.t.sf(abs(t), df)`, where `sf` is
the "survival function" (the area to the right). Build it, then check against `stats.ttest_1samp`.
"""),
        code(r"""
mu0 = 18.0
n = len(oj)
xbar = oj.mean()
s = oj.std(ddof=1)                       # sample SD: divide by n-1 (ddof=1) — essential!
se = s / np.sqrt(n)                      # standard error of the mean

t_one = (xbar - mu0) / se               # the formula, verbatim
df_one = n - 1
p_one = 2 * stats.t.sf(abs(t_one), df_one)   # two-sided: both tails beyond |t|

print(f'OJ group: n = {n}, mean = {xbar:.4f}, s = {s:.4f}')
print(f'by hand : t = {t_one:.6f}, df = {df_one}, p = {p_one:.6f}')

# Check against the library:
res = stats.ttest_1samp(oj, popmean=mu0)
print(f'scipy   : t = {res.statistic:.6f}, df = {res.df}, p = {res.pvalue:.6f}')
print('match?', np.isclose(t_one, res.statistic), np.isclose(p_one, res.pvalue))
"""),
        md(r"""
$t = 2.21$ on $df = 29$, giving $p = 0.0353$ — and our from-scratch numbers match SciPy exactly. The
denominator `s/sqrt(n)` is the only subtle part: it uses the **sample** SD (`ddof=1`), because in real life
we never know the true population spread.

### Two-sample Welch t — two independent groups

Now compare **OJ** against **VC** (different guinea pigs, so *independent* groups). **Welch's** t-test does
**not** assume the two groups share a variance, so each group keeps its own $s$ and $n$:

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}}}.$$

The degrees of freedom are *not* simply $n_1+n_2-2$; Welch uses the **Welch–Satterthwaite** approximation,
which we build below. Then `p = 2 * stats.t.sf(|t|, df)` as before.
"""),
        code(r"""
n1, n2 = len(oj), len(vc)
m1, m2 = oj.mean(), vc.mean()
v1, v2 = oj.var(ddof=1), vc.var(ddof=1)          # sample variances (ddof=1)

# The Welch t: each group contributes its own variance/size to the standard error.
se_welch = np.sqrt(v1 / n1 + v2 / n2)
t_welch = (m1 - m2) / se_welch

# The Welch-Satterthwaite degrees of freedom (build the formula piece by piece):
a = v1 / n1
b = v2 / n2
df_welch = (a + b) ** 2 / (a**2 / (n1 - 1) + b**2 / (n2 - 1))

p_welch = 2 * stats.t.sf(abs(t_welch), df_welch)

print(f'OJ: n = {n1}, mean = {m1:.4f}, s = {np.sqrt(v1):.4f}')
print(f'VC: n = {n2}, mean = {m2:.4f}, s = {np.sqrt(v2):.4f}')
print(f'by hand : t = {t_welch:.6f}, df = {df_welch:.6f}, p = {p_welch:.6f}')

# Check against the library's Welch test:
welch = stats.ttest_ind(oj, vc, equal_var=False)
print(f'scipy   : t = {welch.statistic:.6f}, df = {welch.df:.6f}, p = {welch.pvalue:.6f}')
print('match?', np.isclose(t_welch, welch.statistic),
      np.isclose(df_welch, welch.df), np.isclose(p_welch, welch.pvalue))
"""),
        md(r"""
$t = 1.92$ on a fractional $df = 55.31$ (that decimal df is the Welch–Satterthwaite approximation at work),
giving $p = 0.0606$ — all three numbers match SciPy. Note the p-value is just *above* 0.05: a sizable
sample gap that the test still calls *non-significant*. "Not significant" means "couldn't rule out zero,"
not "no effect."

### Paired t — the key trick: reduce each pair to one difference

The swim data has the **same** swimmer in two columns. The paired t-test is **not** a new formula — it is
the one-sample test applied to the **differences** $d_i = \text{wet}_i - \text{suit}_i$, tested against 0.
Collapse the 12 pairs to 12 differences, then it's exactly the Section-3.1 recipe with $\mu_0 = 0$.
"""),
        code(r"""
d = wet - suit                           # vectorized: one difference per swimmer, all 12 at once
n_d = len(d)

# It's a one-sample t on d against 0 — same three lines as before:
t_paired = d.mean() / (d.std(ddof=1) / np.sqrt(n_d))
df_paired = n_d - 1
p_paired = 2 * stats.t.sf(abs(t_paired), df_paired)

print(f'mean difference = {d.mean():+.4f} m/s   (s = {d.std(ddof=1):.4f}, n = {n_d})')
print(f'by hand : t = {t_paired:.6f}, df = {df_paired}, p = {p_paired:.3e}')

# Check against scipy two ways — ttest_rel on the pairs, and ttest_1samp on the differences:
rel = stats.ttest_rel(wet, suit)
one = stats.ttest_1samp(d, 0.0)
print(f'ttest_rel    : t = {rel.statistic:.6f}, p = {rel.pvalue:.3e}')
print(f'ttest_1samp(d): t = {one.statistic:.6f}, p = {one.pvalue:.3e}   (identical to ttest_rel)')
print('match?', np.isclose(t_paired, rel.statistic), np.isclose(p_paired, rel.pvalue))
"""),
        md(r"""
$t = 12.32$ on $df = 11$, $p \approx 8.9\times10^{-8}$ — a thunderingly significant wetsuit effect, and our
from-scratch `t` matches both `ttest_rel` and `ttest_1samp` on the differences. That equality *is* the whole
idea: **a paired test is a one-sample test on the differences.**

### Why pairing helps — the headline contrast

Now the punchline from the core lesson. What if we *ignore* the pairing and (wrongly) run a two-sample test
on the same two columns, pretending the 12 wetsuit times and 12 swimsuit times are unrelated groups?
"""),
        code(r"""
unpaired = stats.ttest_ind(wet, suit, equal_var=False)   # WRONG here: throws away the pairing

print('SAME 24 numbers, two analyses:')
print(f'  PAIRED   (correct):  t = {rel.statistic:6.2f},  p = {rel.pvalue:.3e}   -> clearly significant')
print(f'  UNPAIRED (wrong):    t = {unpaired.statistic:6.2f},  p = {unpaired.pvalue:.4f}      -> NOT significant')
print()
print(f'p-value blows up from {rel.pvalue:.1e} to {unpaired.pvalue:.3f} just by ignoring the pairing.')
print(f'Why? wetsuit & swimsuit speeds are correlated r = {np.corrcoef(wet, suit)[0,1]:.3f}:')
print('a fast swimmer is fast in BOTH suits, and pairing subtracts that swimmer-to-swimmer spread away.')
"""),
        md(r"""
Same data, opposite verdict: the correct paired test gives $p \approx 9\times10^{-8}$, the wrong unpaired
one gives $p \approx 0.185$. Pairing subtracts away the large *between-swimmer* variation (the speeds are
correlated $r = 0.99$), shrinking the standard error so the same gap looks decisive. **If your two numbers
come from the same unit, pair them.**

## 4) The idiom — vectorized differences, and building the Welch df

### Vectorized differences vs. a loop

The paired test hinges on `d = wet - suit`. NumPy subtracts the two arrays **element by element in one
expression** — no loop over swimmers. You *could* write the loop; it gives the same numbers but is slower
and noisier to read. Compare them.
"""),
        code(r"""
# The loop way (how you might first think of it):
d_loop = np.empty(len(wet))
for i in range(len(wet)):
    d_loop[i] = wet[i] - suit[i]

# The vectorized way (what NumPy is for):
d_vec = wet - suit

print('first 5 differences, loop      :', np.round(d_loop[:5], 3))
print('first 5 differences, vectorized:', np.round(d_vec[:5], 3))
print('same array?', np.allclose(d_loop, d_vec))
print('\nThe vectorized form is one line, reads as the math (after - before), and runs far faster on big data.')
"""),
        md(r"""
### Reading the Welch df formula

The scariest line in Section 3 was the Welch–Satterthwaite degrees of freedom. It is just a weighted
combination of the two groups' variance contributions:

$$df = \frac{\left(\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}\right)^2}
            {\dfrac{(s_1^2/n_1)^2}{n_1-1} + \dfrac{(s_2^2/n_2)^2}{n_2-1}}.$$

Naming the two pieces `a = s1**2/n1` and `b = s2**2/n2` (which is exactly what we did) turns that wall of
symbols into one readable line. Watch how it lands *between* the smallest group's $n-1$ and the pooled
$n_1+n_2-2$.
"""),
        code(r"""
a = v1 / n1
b = v2 / n2
df_welch = (a + b) ** 2 / (a**2 / (n1 - 1) + b**2 / (n2 - 1))

print(f'smallest group df (n-1)     : {min(n1, n2) - 1}')
print(f'Welch-Satterthwaite df      : {df_welch:.4f}   <- a fractional df, not an integer')
print(f'pooled / equal-var df (n1+n2-2): {n1 + n2 - 2}')
print('\nWelch df sits between those bounds — it is the equal-variance df discounted for unequal spreads.')
"""),
        md(r"""
## 5) Refactor into reusable functions

You've now written the same handful of lines a few times. The programmer's reflex is to **wrap repeated work
in a function**: name it, give it inputs, `return` the result. Then "run a one-sample t-test" or "run a
Welch test" is one call you can reuse on *any* data — and a paired test is just `t_test_one_sample` on the
differences, with no new code at all.
"""),
        code(r"""
def t_test_one_sample(x, mu0=0.0):
    '''Return (t, df, two-sided p) for a one-sample t-test of mean(x) vs mu0.'''
    x = np.asarray(x, dtype=float)
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)        # sample SD -> standard error
    t = (x.mean() - mu0) / se
    df = n - 1
    p = 2 * stats.t.sf(abs(t), df)
    return t, df, p


def welch_t(a, b):
    '''Return (t, df, two-sided p) for Welch's two-sample t-test (unequal variances).'''
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb     # each group's variance contribution
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return t, df, p


# Reuse them on all three tests — no copy-pasted formulas:
print('one-sample (OJ vs 18) :', tuple(round(v, 6) for v in t_test_one_sample(oj, 18.0)))
print('Welch      (OJ vs VC) :', tuple(round(v, 6) for v in welch_t(oj, vc)))
print('paired (one-sample on d):', tuple(round(v, 6) for v in t_test_one_sample(wet - suit, 0.0)))
"""),
        code(r"""
# Confirm the reusable functions still agree with scipy on every test:
t1, df1, p1 = t_test_one_sample(oj, 18.0)
tw, dfw, pw = welch_t(oj, vc)
tp, dfp, pp = t_test_one_sample(wet - suit, 0.0)

print('one-sample match? ', np.isclose(t1, stats.ttest_1samp(oj, 18.0).statistic),
      np.isclose(p1, stats.ttest_1samp(oj, 18.0).pvalue))
print('Welch      match? ', np.isclose(tw, stats.ttest_ind(oj, vc, equal_var=False).statistic),
      np.isclose(pw, stats.ttest_ind(oj, vc, equal_var=False).pvalue))
print('paired     match? ', np.isclose(tp, stats.ttest_rel(wet, suit).statistic),
      np.isclose(pp, stats.ttest_rel(wet, suit).pvalue))
"""),
        md(r"""
A function is a contract: *give me data, I'll give you `(t, df, p)`.* Once it's correct, you stop thinking
about the formulas and start thinking in **groups and comparisons** — the leap from "running statistics" to
"doing statistics in code." (And yes, `scipy.stats.ttest_*` exists; the point was to *build* it so you know
what it's doing.)
"""),
        md(r"""
## 6) Now you code

Write the code yourself. Add a new cell (the **+** button in the toolbar) and try each — predict the answer
before you run it.

1. **A one-sided p-value.** Our `t_test_one_sample` returns a *two-sided* p (it doubled the tail). For the OJ
   group vs 18, the alternative "the mean is *greater* than 18" needs just the **right** tail:
   `stats.t.sf(t, df)` with no factor of 2. Compute it for `t_test_one_sample(oj, 18.0)` and confirm it's
   exactly **half** the two-sided p.

2. **Pooled (equal-variance) two-sample t.** Write `pooled_t(a, b)` using the equal-variance formula:
   $s_p = \sqrt{\dfrac{(n_a-1)s_a^2 + (n_b-1)s_b^2}{n_a+n_b-2}}$, then $t = (\bar a - \bar b)/(s_p\sqrt{1/n_a + 1/n_b})$,
   with $df = n_a+n_b-2$. Test `pooled_t(oj, vc)` against `stats.ttest_ind(oj, vc, equal_var=True)` (note
   `equal_var=True`, the *pooled* version). It should match.

3. **Paired without the helper.** Build the paired t for the swim data in three lines *without* calling
   `t_test_one_sample`: compute `d = wet - suit`, then `t = d.mean() / (d.std(ddof=1)/np.sqrt(len(d)))`, then
   the p-value. Check it equals `stats.ttest_rel(wet, suit)`.

4. **A different group and target.** The strongest tooth-growth effect is at the highest dose. Pull
   `oj2 = tooth[(tooth['supp']=='OJ') & (tooth['dose']==2.0)]['len'].to_numpy()` and
   `vc2 = tooth[(tooth['supp']=='VC') & (tooth['dose']==2.0)]['len'].to_numpy()`, then run
   `welch_t(oj2, vc2)`. Predict first: at dose 2.0 the two delivery methods are famously almost identical —
   does your from-scratch `t` come out near zero?

## What you learned

**Coding**
- The t-statistic is **one formula** — `(estimate - null) / standard_error` — and the only moving parts are
  what you put in the numerator and denominator. You wrote all three tests from it.
- A two-sided p-value is `2 * stats.t.sf(abs(t), df)`; `sf` is the right-tail area, and you double it for
  *both* tails. Halve it for a one-sided test.
- **Vectorization** — `d = wet - suit` subtracts two whole arrays in one expression — replaces the loop,
  reads as the math, and runs far faster.
- **Boolean masks** (`tooth.loc[tooth['supp']=='OJ', 'len']`) pull one group's response out of a tidy table.
- **Refactor repeated work into functions** (`t_test_one_sample`, `welch_t`) with clear inputs and a
  `return`, then reuse them — the paired test needed *no* new function, just `t_test_one_sample(wet-suit)`.

**Statistics (now demystified)**
- The **one-sample** `t = (\bar{x}-\mu_0)/(s/\sqrt{n})` on $df=n-1$: you got $t=2.21$, $p=0.0353$, matching
  `ttest_1samp` exactly.
- The **Welch two-sample** `t` keeps each group's own $s^2/n$ and uses the **Welch–Satterthwaite** $df$ (a
  fractional $55.31$, not an integer): $t=1.92$, $p=0.0606$, matching `ttest_ind(equal_var=False)`.
- The **paired** test is *literally* a one-sample t on the differences $d=\text{wet}-\text{suit}$ against 0
  ($t=12.32$, $p\approx9\times10^{-8}$) — `ttest_rel` and `ttest_1samp(d)` give the identical number.
- **Pairing beats ignoring it:** the correct paired test ($p\approx9\times10^{-8}$) detected an effect the
  wrong unpaired test ($p\approx0.185$) missed, because pairing subtracts away large between-subject
  variation (speeds correlated $r=0.99$) and shrinks the standard error.
- Everywhere, a **sample** SD uses `ddof=1` (divide by $n-1$); NumPy's default `ddof=0` would make every `t`
  come out a hair too big and stop matching SciPy.

**Back to the core lesson** ([Lesson 15: t-tests: one-sample, two-sample & paired](15-t-tests.ipynb)) for the
*meaning* of these numbers, or read on through the course.
"""),
        md(nav_md(ID)),
    ]


if __name__ == "__main__":
    save_lesson(ID, cells())
