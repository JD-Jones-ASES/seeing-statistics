r"""Lesson 14 (code variant) — "The code behind it": build Lesson 14's proportion
inference from counts in Python, on the SAME US Births 2014 data + real example as
the core lesson. Count successes with a boolean mask (mean of a 0/1 column = the
proportion), build p-hat, its standard error, a Wald CI and a one-proportion
z-test straight from the formulas, then the two-proportion difference (unpooled SE
for the CI, pooled SE for the test) — every result checked against statsmodels
with np.isclose printing "match? True" — and finally wrap it as reusable
prop_ci() and two_prop_ztest() functions. Follows the Phase-4 code-variant
skeleton set by lesson_01_code.py. See MANIFEST in lessonkit.py."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # tools/
from lessonkit import md, code, save_lesson, header_md, nav_md

ID = "proportions-code"

INTRO = r"""
**In Lesson 14 you estimated proportions, wrapped them in confidence intervals, and ran z-tests** — and you
*earned* the formulas by simulating sampling distributions and watching `statsmodels` agree. This companion
**opens the hood on the formulas themselves**: you'll build $\hat{p}$, its standard error, the CI and both
z-tests *from counts you compute yourself*, so a "proportion z-test" stops being a library call and becomes a
few lines you could write from memory.

Same 1,000 US births, same low-birth-weight and smoking example, same final numbers — but the focus shifts
from *what the inference means* to **how you compute it in Python from a raw column**. We'll go in the order a
programmer actually works:

1. **The goal** — what we're building.
2. **The data in code** — load it and *count successes* with a boolean mask (the mean of a 0/1 column is the proportion).
3. **Build it step by step** — turn the formulas for $\hat{p}$, SE, the CI and the z-tests into NumPy, and check each against `statsmodels`.
4. **The idiom** — boolean-mask counting, building an SE from its formula, and pooled vs. unpooled SE.
5. **Refactor into functions** — wrap it as your own reusable `prop_ci()` and `two_prop_ztest()`.
6. **Now you code** — predict-then-run exercises.
7. **What you learned** — the coding *and* the statistics.

You don't need Lesson 14 fresh in mind, but it helps. Run each grey cell with **Shift + Enter**, top to bottom.
"""


def cells():
    return [
        md(header_md(ID, intro=INTRO)),
        md(r"""
## 1) The goal

By the end of this notebook you'll have written, from the formulas, the whole proportion-inference toolkit the
core lesson handed you: a sample proportion $\hat{p}$, its **standard error**, a 95% **confidence interval**, a
**one-proportion z-test** against a benchmark, and the **two-proportion** difference, CI and z-test — each one
checked against `statsmodels` so you *see* your hand calculation land on the library's answer. Nothing here
needs new statistics; everything is the Python *underneath* the one-liners.

The tools we'll use, and what each is for:

| Tool | What it is | What we use it for |
|---|---|---|
| **pandas** | tables (`DataFrame`) and columns (`Series`) | loading the CSV, the boolean mask, the 2×2 crosstab |
| **NumPy** | fast math on whole arrays at once | counting (`.sum()`, `.mean()`), the SE formula, `np.isclose` |
| **SciPy** | distributions | `stats.norm.ppf` for $z^{*}$, `stats.norm.sf` for p-values |
| **statsmodels** | ready-made stats routines | the trusted answer we check ourselves against |

The single most important idea below is plain: a **"success" is a yes/no event we're counting** (here, a
low-birth-weight baby), and once we turn that into a column of **1s and 0s**, *counting* successes is
`.sum()` and the *proportion* of successes is just `.mean()`. Everything else is one formula at a time.
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
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
sl.use_course_style()
rng = np.random.default_rng(0)   # fixed seed -> reproducible (we barely use randomness here)

# The SAME data as the core lesson: 1,000 US births from 2014.
BIRTHS_URL = 'https://www.openintro.org/data/csv/births14.csv'
births = sl.load_csv('births14.csv', url=BIRTHS_URL)
print(f'Loaded a {type(births).__name__} with shape {births.shape}  (rows, columns)')
"""),
        md(r"""
## 2) The data in code

Before any inference, a programmer *interrogates the object*: how big is it, what columns matter, what exact
labels do they hold? We'll lean on two columns: **`lowbirthweight`** (the text label `'low'` / `'not low'`)
and **`habit`** (the mother's smoking status). Check the label strings *first* — your whole count depends on
matching them exactly.
"""),
        code(r"""
print('shape :', births.shape)
# The exact label strings we must match (a typo here would silently miscount):
print('lowbirthweight labels:', list(births['lowbirthweight'].dropna().unique()))
print('habit labels         :', list(births['habit'].dropna().unique()))
births[['weight', 'weeks', 'habit', 'lowbirthweight']].head()
"""),
        md(r"""
### Counting successes with a boolean mask

Here is the move the whole lesson rests on. A **"success"** is simply *the event we're counting* — here, a
baby labelled `'low'` (no value judgement implied). The comparison `births['lowbirthweight'] == 'low'` makes a
column of `True`/`False` — a **boolean mask**. Two facts about that mask do all the work:

- **`.sum()`** counts the `True`s, because Python treats `True` as `1` and `False` as `0`. That's the number
  of successes, $x$.
- **`.mean()`** of a 0/1 column is `(number of 1s) / (total)` — which is *exactly* the proportion $\hat{p}$.

So `mask.sum()` and `mask.mean()` give you $x$ and $\hat{p}$ with no separate counting code at all.
"""),
        code(r"""
mask = (births['lowbirthweight'] == 'low')     # a True/False column: True where the baby is low-birth-weight

n    = len(mask)            # sample size (no missing labels here, so len is fine)
x    = int(mask.sum())      # successes: .sum() counts the Trues  (True == 1)
phat = mask.mean()          # proportion: the MEAN of a 0/1 column IS the proportion of 1s

print(f'sample size        n     = {n}')
print(f'successes (low)    x     = {x}    <- mask.sum() counts the Trues')
print(f'proportion         p-hat = {phat:.4f}    <- mask.mean() of a 0/1 column')
print(f'sanity: x / n            = {x / n:.4f}    <- same thing, computed the long way')
print('\nSo "count the successes" is .sum(), and "what fraction are successes" is .mean(). One column, two reads.')
"""),
        md(r"""
## 3) Build it step by step

Now the heart of it. Each quantity is a short formula; we'll write the formula in NumPy and then check it
against `statsmodels` with `np.isclose`, printing **match?**. Once you've built it, the library call is no
longer a mystery box.

### The standard error of $\hat{p}$

The wobble of $\hat{p}$ from sample to sample is the **standard error**:

$$\text{SE}(\hat{p}) = \sqrt{\dfrac{\hat{p}\,(1-\hat{p})}{n}}.$$

In code that is one line — `np.sqrt(phat * (1 - phat) / n)`. (The core lesson *proved* this formula by
simulating the sampling distribution; here we just take it and build on it.)
"""),
        code(r"""
se = np.sqrt(phat * (1 - phat) / n)        # the SE formula, verbatim
print(f'SE(p-hat) = sqrt(p-hat*(1-p-hat)/n) = {se:.4f}')

# A quick safety check the core lesson made: the normal approx. needs >=10 of each.
print(f'n*p-hat = {n*phat:.0f}  and  n*(1-p-hat) = {n*(1-phat):.0f}  -> both >= 10, normal recipe is safe.')
"""),
        md(r"""
### A 95% confidence interval (the Wald interval)

Because $\hat{p}$ is approximately normal, the interval is the familiar **estimate ± margin**:

$$\hat{p} \;\pm\; z^{*}\,\text{SE}, \qquad z^{*} = \texttt{stats.norm.ppf(0.975)} \approx 1.96 \text{ for 95\%}.$$

`stats.norm.ppf(0.975)` is the **inverse normal** — "the z below which 97.5% of the curve sits" — which leaves
2.5% in each tail for a 95% interval. Build it, then check it against `proportion_confint` from statsmodels.
"""),
        code(r"""
zstar  = stats.norm.ppf(0.975)             # 1.96 for 95%
margin = zstar * se
lo, hi = phat - margin, phat + margin
print(f'z* for 95%      = {zstar:.4f}')
print(f'margin of error = {margin:.4f}')
print(f'95% CI by hand  = ({lo:.4f}, {hi:.4f})   ->  ({lo*100:.1f}%, {hi*100:.1f}%)')

# statsmodels computes the very same "Wald"/normal interval from the counts:
lo_sm, hi_sm = proportion_confint(count=x, nobs=n, alpha=0.05, method='normal')
print(f'statsmodels CI  = ({lo_sm:.4f}, {hi_sm:.4f})')
print('match?', np.isclose(lo, lo_sm) and np.isclose(hi, hi_sm))
"""),
        md(r"""
> **Small-sample caveat (one line).** This $\hat{p} \pm z^{*}\,\text{SE}$ interval is the **Wald** interval; when
> successes are scarce it can be a touch too optimistic, so sturdier variants — **Wilson** and
> **Agresti–Coull** — exist (`proportion_confint(..., method='wilson')`). With 81 successes here they barely
> differ, but it's good to know the safer tool is one keyword away.

### A one-proportion z-test against a benchmark

Suppose a report claims the national low-birth-weight rate is $p_0 = 0.08$. The test asks whether our sample is
consistent with that. The twist: we **assume the null is true** while testing it, so the SE is built from
$p_0$, **not** from $\hat{p}$:

$$z = \dfrac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}}, \qquad
p\text{-value} = 2\,\texttt{stats.norm.sf}(|z|).$$

`stats.norm.sf(|z|)` is the **upper-tail area** (the "survival function," $1-\text{CDF}$); doubling it makes the
test **two-sided**. Check against statsmodels — passing `prop_var=p0` tells it to use the null value in the SE.
"""),
        code(r"""
p0  = 0.08                                 # the benchmark we test against
se0 = np.sqrt(p0 * (1 - p0) / n)           # SE UNDER THE NULL: built from p0, not p-hat
z   = (phat - p0) / se0
pval = 2 * stats.norm.sf(abs(z))           # sf = upper tail; doubled -> two-sided
print(f'z-statistic         = {z:.4f}')
print(f'p-value (two-sided) = {pval:.4f}')

# statsmodels: value=p0 is the benchmark, prop_var=p0 forces the null SE (the correct test SE).
z_sm, p_sm = proportions_ztest(count=x, nobs=n, value=p0, prop_var=p0)
print(f'statsmodels: z = {z_sm:.4f}, p = {p_sm:.4f}')
print('match?', np.isclose(z, z_sm) and np.isclose(pval, p_sm))
print(f'\n-> p = {pval:.2f} > 0.05: our data is fully consistent with an 8% rate (we do NOT reject H0).')
"""),
        md(r"""
### Two proportions: smokers vs. nonsmokers

Now the real question — is the low-birth-weight rate different for mothers who smoke? Start, as always, by
*counting*. A boolean mask works per-group too: slice the table by `habit`, then take `.sum()` and `len(...)`
in each group. We drop the few rows with unknown smoking status first, exactly as the core lesson does.
"""),
        code(r"""
b = births.dropna(subset=['habit']).copy()
is_low = (b['lowbirthweight'] == 'low')          # the success mask again, on the cleaned table

# Group 1 = smokers, group 2 = nonsmokers. Count successes and sizes with masks.
smoker = (b['habit'] == 'smoker')
x1 = int(is_low[smoker].sum());   n1 = int(smoker.sum())          # successes & size, smokers
x2 = int(is_low[~smoker].sum());  n2 = int((~smoker).sum())       # ~smoker = the nonsmokers
p1, p2 = x1 / n1, x2 / n2
diff = p1 - p2

print(f'(dropped {len(births) - len(b)} births with unknown smoking status)')
print(f'smokers    : {x1:>3} low of {n1:>3}   ->  p1-hat = {p1:.4f}  ({p1*100:.1f}%)')
print(f'nonsmokers : {x2:>3} low of {n2:>3}   ->  p2-hat = {p2:.4f}  ({p2*100:.1f}%)')
print(f'observed difference  p1 - p2 = {diff:.4f}   ({diff*100:.1f} percentage points)')
"""),
        md(r"""
### Unpooled SE for the CI; pooled SE for the test

This is the one genuinely subtle point — and it's a *coding* point as much as a statistics one: the CI and the
test use **different standard errors**.

- **Confidence interval** — there is no assumption that the rates are equal, so each group contributes its
  **own** variance; we *add* them (independent groups). This is the **unpooled** SE:
  $$\text{SE}_{\text{unpooled}} = \sqrt{\dfrac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \dfrac{\hat{p}_2(1-\hat{p}_2)}{n_2}}.$$
- **z-test** — the null says the rates are **equal**, so the best estimate of that one shared rate **pools**
  both groups, $\hat{p}_{\text{pool}} = (x_1+x_2)/(n_1+n_2)$, giving the **pooled** SE:
  $$\text{SE}_{\text{pooled}} = \sqrt{\hat{p}_{\text{pool}}(1-\hat{p}_{\text{pool}})\left(\tfrac{1}{n_1}+\tfrac{1}{n_2}\right)}.$$

First the CI with the **unpooled** SE.
"""),
        code(r"""
se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)   # UNPOOLED: each group's own variance, added
margin_d = zstar * se_diff
lo_d, hi_d = diff - margin_d, diff + margin_d
print(f'unpooled SE of the difference = {se_diff:.4f}')
print(f'95% CI for p1 - p2 = ({lo_d:.4f}, {hi_d:.4f})')
print(f'                   = ({lo_d*100:.1f}, {hi_d*100:.1f}) percentage points')
print('\nThe whole interval is ABOVE 0, so 0 (no difference) is not plausible: smokers have a higher rate.')
"""),
        md(r"""
Now the **two-proportion z-test** with the **pooled** SE, $z = (\hat{p}_1 - \hat{p}_2)/\text{SE}_{\text{pooled}}$.
We check it against `proportions_ztest(count=[x1, x2], nobs=[n1, n2])` — and here's the nice part: that
statsmodels call **pools by default**, which is exactly what our test SE does, so the two should match to many
decimals.
"""),
        code(r"""
p_pool  = (x1 + x2) / (n1 + n2)                              # one shared rate, under H0
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) # POOLED SE for the test
z2   = diff / se_pool
pval2 = 2 * stats.norm.sf(abs(z2))
print(f'pooled p-hat        = {p_pool:.4f}')
print(f'z-statistic         = {z2:.4f}')
print(f'p-value (two-sided) = {pval2:.3e}')

# statsmodels' two-proportion test pools the groups too -- so it should agree:
z_sm2, p_sm2 = proportions_ztest(count=[x1, x2], nobs=[n1, n2])
print(f'statsmodels: z = {z_sm2:.4f}, p = {p_sm2:.3e}')
print('match?', np.isclose(z2, z_sm2) and np.isclose(pval2, p_sm2))
print(f'\n-> p = {pval2:.1e} << 0.05: the smoker/nonsmoker gap is real, not sampling luck.')
"""),
        md(r"""
## 4) The idiom

Three small habits carried this whole notebook. Worth stating plainly, because you'll reuse them constantly.

### Boolean-mask counting — `.sum()` and `.mean()`

A `True`/`False` column is secretly a column of `1`s and `0`s. So:

- **`mask.sum()`** = how many are `True` = the **count of successes** ($x$).
- **`mask.mean()`** = the fraction that are `True` = the **proportion** ($\hat{p}$).

No loop, no counter variable. And it composes: `is_low[smoker].sum()` counts successes *within a group*.
"""),
        code(r"""
demo = pd.Series([True, False, True, True, False])
print('the mask        :', demo.to_list())
print('demo.sum()      =', demo.sum(),  '  <- count of Trues (successes)')
print('demo.mean()     =', demo.mean(), '  <- fraction True == the proportion')
print('len(demo)       =', len(demo),   '  <- the denominator n')
print('\nThat is the entire trick behind x, n and p-hat above.')
"""),
        md(r"""
### Building a standard error from its formula, and pooled vs. unpooled

Every SE in this lesson is the *same shape*: a variance term under a `np.sqrt`. The only choice is **which
variance**:

- **one proportion:** `phat*(1-phat)/n`
- **difference, unpooled (CI):** add each group's own `p*(1-p)/n`
- **difference, pooled (test):** one shared `p_pool*(1-p_pool)` times `(1/n1 + 1/n2)`
- **one-proportion test:** the null version, `p0*(1-p0)/n`

Same `sqrt`, different inside. Lining them up makes the pooled-vs-unpooled distinction concrete — and shows
why the CI and the test, built on different SEs, are answering slightly different questions.
"""),
        code(r"""
print(f'one-proportion SE   (uses p-hat) : {np.sqrt(phat*(1-phat)/n):.4f}')
print(f'one-prop TEST SE    (uses p0)    : {np.sqrt(p0*(1-p0)/n):.4f}')
print(f'difference, UNPOOLED (for the CI): {np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2):.4f}')
print(f'difference, POOLED   (for the z) : {np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2)):.4f}')
print('\nSame sqrt-of-a-variance shape every time; only the variance inside changes.')
"""),
        md(r"""
## 5) Refactor into reusable functions

You've now written the CI math and the two-proportion test math once each. The programmer's reflex is to
**wrap repeated work in a function**: name it, give it inputs, `return` the result. Then "confidence interval
for a proportion" and "two-proportion z-test" are each one call you can reuse on *any* counts — these births,
a different split, a different dataset entirely.
"""),
        code(r"""
def prop_ci(successes, n, conf=0.95):
    '''Wald (normal) confidence interval for one proportion, from raw counts.
    Returns (lo, hi). Same recipe as Section 3: p-hat +/- z* * SE.
    '''
    phat = successes / n
    se = np.sqrt(phat * (1 - phat) / n)
    zstar = stats.norm.ppf(1 - (1 - conf) / 2)        # 0.975 for conf=0.95
    return phat - zstar * se, phat + zstar * se


def two_prop_ztest(s1, n1, s2, n2):
    '''Two-sided two-proportion z-test with the POOLED SE (the standard test).
    Returns (z, p). Same recipe as Section 3.
    '''
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, 2 * stats.norm.sf(abs(z))


# Reuse them on the real data, and re-confirm against statsmodels:
ci_lo, ci_hi = prop_ci(x, n, conf=0.95)
lo_sm, hi_sm = proportion_confint(count=x, nobs=n, alpha=0.05, method='normal')
print(f'prop_ci(x, n)            = ({ci_lo:.4f}, {ci_hi:.4f})   match statsmodels? '
      f'{np.isclose(ci_lo, lo_sm) and np.isclose(ci_hi, hi_sm)}')

zf, pf = two_prop_ztest(x1, n1, x2, n2)
zsm, psm = proportions_ztest(count=[x1, x2], nobs=[n1, n2])
print(f'two_prop_ztest(...)      = (z={zf:.4f}, p={pf:.3e})   match statsmodels? '
      f'{np.isclose(zf, zsm) and np.isclose(pf, psm)}')
"""),
        md(r"""
A function is a contract: *give me counts, I'll give you an interval (or a z and a p).* Once it's correct, you
stop thinking about the `sqrt` and the `ppf` and start thinking in **counts, intervals and tests** — exactly
the leap from "running a proportion test" to "doing proportion inference in code." (And yes, `statsmodels`
already ships these; the point was to *build* them so you know what's inside.)
"""),
        md(r"""
## 6) Now you code

Write the code yourself. Add a new cell (the **+** button in the toolbar) and try each — predict the answer
before you run it.

1. **Count from a mask.** Without using `proportion_confint`, recompute $x$, $n$ and $\hat{p}$ for the
   *premature* babies: build `mask = births['premie'] == 'premie'` (check the labels with
   `births['premie'].dropna().unique()` first), then `mask.sum()`, `len(mask)` and `mask.mean()`. Confirm
   `mask.mean()` equals `mask.sum() / len(mask)`.

2. **A 99% interval is wider.** Call `prop_ci(x, n, conf=0.99)` and compare it to the 95% interval. Predict
   *before running*: does raising the confidence make the interval **wider** or **narrower**? Check your
   `prop_ci` against `proportion_confint(x, n, alpha=0.01, method='normal')`.

3. **A different two-group split.** Swap `habit` for **`mature`** (labels `'mature mom'` / `'younger mom'`).
   Recompute `x1, n1, x2, n2` with masks the same way, then call `two_prop_ztest(x1, n1, x2, n2)`. Is the
   low-birth-weight rate significantly different for mature vs. younger mothers? Verify with
   `proportions_ztest([x1, x2], [n1, n2])`.

4. **Wald vs. Wilson.** Build the one-proportion CI two ways with statsmodels —
   `proportion_confint(x, n, method='normal')` (the Wald interval you coded) and
   `proportion_confint(x, n, method='wilson')`. With 81 successes, how far apart are they? Now imagine only
   **3** successes out of 30 (`proportion_confint(3, 30, method=...)`) — does the gap between Wald and Wilson
   grow?

## What you learned

**Coding**
- A **boolean mask** (`col == value`) is a column of `True`/`False`; because `True == 1`, **`mask.sum()` counts
  successes** and **`mask.mean()` *is* the proportion** — no loop, no counter. It composes across groups:
  `is_low[smoker].sum()`.
- An **SE is built from its formula** in one line: a variance term under `np.sqrt`. Only the variance inside
  changes between the one-proportion, unpooled-difference, and pooled-difference cases.
- `stats.norm.ppf(0.975)` gives the multiplier $z^{*}$; `stats.norm.sf(|z|)` (doubled) gives a two-sided
  p-value. Use **`np.isclose`** to confirm your hand calculation matches a library — the programmer's proof.
- **Refactor repeated work into functions** with clear inputs and a `return` (`prop_ci`, `two_prop_ztest`),
  then reuse them on any counts.

**Statistics (now demystified)**
- $\hat{p} = x/n$ and $\text{SE} = \sqrt{\hat{p}(1-\hat{p})/n}$; a 95% CI is $\hat{p} \pm z^{*}\,\text{SE}$ (the
  **Wald** interval — Wilson/Agresti–Coull are sturdier when successes are scarce). You matched statsmodels to
  the decimal.
- A **one-proportion z-test** builds its SE from the **null** $p_0$, not from $\hat{p}$; here $p = 0.91$, so
  the data is consistent with an 8% rate.
- For **two** proportions the **CI uses the unpooled SE** (each group's own variance, added) while the **test
  uses the pooled SE** (one shared rate under $H_0$) — and statsmodels' two-proportion test pools too, so it
  agreed. The smoker/nonsmoker gap (about 21% vs. 6%) is real, with $z \approx 5.5$, $p \approx 4\times10^{-8}$.

**Back to the core lesson** ([Lesson 14: Inference for proportions (one & two)](14-inference-proportions.ipynb))
for the *meaning* of these numbers, or read on through the course.
"""),
        md(nav_md(ID)),
    ]


if __name__ == "__main__":
    save_lesson(ID, cells())
