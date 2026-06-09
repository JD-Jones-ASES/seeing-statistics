r"""Lesson 17 (code variant) — "The code behind it": build the chi-square test of
independence from scratch in Python (NumPy/pandas/SciPy), on the same Titanic
table as the core lesson. Skeleton follows lesson_01_code.py: the goal -> the
data in code -> build it step by step -> the idiom (np.outer / vectorized) ->
refactor into a reusable function -> now you code -> what you learned. See
MANIFEST in lessonkit.py."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # tools/
from lessonkit import md, code, save_lesson, header_md, nav_md

ID = "chi-square-code"

INTRO = r"""
**In Lesson 17 you ran `scipy.stats.chi2_contingency(table)`** and read off a $\chi^2$, a df and a
p-value — the test of independence in one line. This companion **opens the hood**: you'll build that
whole test *yourself*, so "chi-square" stops being a function you call and becomes a handful of array
operations you could write from memory.

Same Titanic table, same final numbers — but the focus shifts from *what the test means* to **how you
compute it in Python**. We'll go in the order a programmer actually works:

1. **The goal** — what we're building.
2. **The data in code** — load it, and turn two text columns into a counts table with `pd.crosstab`.
3. **Build it step by step** — expected counts, the $\chi^2$ statistic, df and p-value, checked against `scipy`.
4. **The idiom** — `np.outer` / broadcasting for the expected table, and the fully vectorized $(O-E)^2/E$ sum.
5. **Refactor into a function** — wrap it all up as your own reusable `chi_square_independence()`.
6. **Now you code** — write-it-yourself exercises.
7. **What you learned** — the coding *and* the statistics.

You don't need Lesson 17 fresh in mind, but it helps. Run each grey cell with **Shift + Enter**, top to bottom.
"""


def cells():
    return [
        md(header_md(ID, intro=INTRO)),
        md(r"""
## 1) The goal

By the end of this notebook you'll have written a function that takes a contingency table of counts and
returns its chi-square statistic, degrees of freedom and p-value — the same three numbers
`scipy.stats.chi2_contingency` hands you for free. Nothing here needs new statistics; everything is the
Python *underneath* the one-liner you already ran.

The tools we'll use, and what each is for:

| Tool | What it is | What we use it for |
|---|---|---|
| **pandas** | tables (`DataFrame`) and columns (`Series`) | loading the CSV, building the counts table with `crosstab` |
| **NumPy** | fast math on whole arrays at once | the actual formulas (`np.outer`, sums over the table) |
| **SciPy** | `scipy.stats` statistical tests | the trusted answer we check ourselves against |

The one move the whole test rests on: compare the counts we **observed** ($O$) to the counts we'd
**expect** under "no relationship" ($E$), and add up the squared, scaled gaps:

$$\chi^2 \;=\; \sum_{\text{cells}} \frac{(O - E)^2}{E}.$$

Every piece of that — the table $O$, the expected table $E$, the sum — is a short array expression. Let's build them.
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

TITANIC_URL = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
titanic = sl.load_csv('titanic.csv', url=TITANIC_URL)
print(f'Loaded a {type(titanic).__name__} with shape {titanic.shape}  (rows, columns)')
"""),
        md(r"""
## 2) The data in code

Before any statistics, a programmer *interrogates the object*: how big is it, what are the columns, what
do the values look like? The two columns we'll test for a relationship — **`Sex`** and **`Survived`** —
are both **categorical** (text, or a 0/1 label), and both are complete (no missing values), so no
`dropna` is needed here.

- `.shape` → `(rows, columns)` as a tuple.
- `.unique()` → the distinct values in a column — the *categories*.
- `.head()` → the first few rows, to eyeball it.
"""),
        code(r"""
print('shape          :', titanic.shape)
print('Survived values:', sorted(titanic['Survived'].unique()), '  (0 = died, 1 = survived)')
print('Sex values     :', titanic['Sex'].unique().tolist())
print('Pclass values  :', sorted(titanic['Pclass'].unique()), '  (1 = first, 2 = second, 3 = third)')
titanic[['Survived', 'Pclass', 'Sex', 'Age']].head()
"""),
        md(r"""
### From two columns to a counts table — `pd.crosstab`

A test of independence needs the **contingency table**: a grid that counts how many passengers fall into
each (`Sex`, `Survived`) combination. `pd.crosstab(var1, var2)` does exactly that — one row per category
of the first variable, one column per category of the second, each cell a count.

We'll do the math on the table's raw numbers, `O.values` (a plain NumPy array), so the array operations
(`np.outer`, broadcasting) stay clean. The row and column **totals** — the *marginals* — are just sums
along each axis, and they're the only ingredients the expected counts need.
"""),
        code(r"""
O = pd.crosstab(titanic['Sex'], titanic['Survived'])
O.columns = ['died (0)', 'survived (1)']
print('OBSERVED counts (Sex x Survived):\n')
print(O)

row_totals = O.sum(axis=1).to_numpy()    # passengers per sex     (sum across columns)
col_totals = O.sum(axis=0).to_numpy()    # passengers per outcome (sum down rows)
grand_total = O.values.sum()

print('\nRow totals (by sex)     :', row_totals.tolist())
print('Column totals (outcome) :', col_totals.tolist())
print('Grand total             :', grand_total)
"""),
        md(r"""
## 3) Build it step by step

Now the heart of it. Each piece is a short formula — we'll write it in NumPy and check the whole test
against `scipy.stats.chi2_contingency` at the end. Once you've built it, the one-liner is no longer a
mystery box.

### Expected counts under independence

The null hypothesis is **independence**: survival has *nothing* to do with sex. If that were true, the
expected count in each cell is

$$E_{ij} = \frac{(\text{row } i \text{ total}) \times (\text{column } j \text{ total})}{\text{grand total}}.$$

For now we'll build $E$ the literal, spelled-out way — a loop over cells — so the formula is unmistakable.
(In Section 4 we replace this loop with a one-line outer product.)
"""),
        code(r"""
n_rows, n_cols = O.shape
E = np.zeros((n_rows, n_cols))
for i in range(n_rows):
    for j in range(n_cols):
        E[i, j] = row_totals[i] * col_totals[j] / grand_total

expected_df = pd.DataFrame(E.round(2), index=O.index, columns=O.columns)
print('EXPECTED counts if survival were independent of sex:\n')
print(expected_df)
print('\nOBSERVED counts (what actually happened):\n')
print(O)
print('\nFemales survived FAR more than independence predicts (233 vs ~121);')
print('males survived FAR less (109 vs ~221). That gap is what chi^2 will measure.')
"""),
        md(r"""
A built-in sanity check on $E$: because each expected cell is `row_total * col_total / grand_total`, the
expected table has the **same row totals and column totals as the observed one** — independence only
redistributes the counts *within* those fixed margins, it never changes them. If your margins don't match,
your $E$ is wrong.
"""),
        code(r"""
print('row totals  O:', O.values.sum(axis=1).tolist(), '   E:', E.sum(axis=1).round(2).tolist())
print('col totals  O:', O.values.sum(axis=0).tolist(), '   E:', E.sum(axis=0).round(2).tolist())
print('grand total O:', O.values.sum(),               '          E:', round(E.sum(), 2))
print('\nSame margins -> our expected table is consistent with the observed one.')
"""),
        md(r"""
### The statistic, the df, and the p-value

Now turn the formula into code. `O.values - E` subtracts the two whole tables **element-by-element**;
`** 2` squares every cell; `/ E` scales each by its expected count; `.sum()` adds all the pieces — no
loop over cells. Then:

- **degrees of freedom:** $\text{df} = (\text{rows}-1)\times(\text{columns}-1)$.
- **p-value:** the area to the right of our $\chi^2$ under the chi-square curve, `stats.chi2.sf(chi2, df)`
  (`sf` is the *survival function*, $1 -$ CDF — the right tail).
"""),
        code(r"""
chi2 = np.sum((O.values - E) ** 2 / E)        # the formula, vectorized over the whole table
df = (n_rows - 1) * (n_cols - 1)
p = stats.chi2.sf(chi2, df)

print(f'chi^2 by hand = {chi2:.4f}')
print(f'df = (rows - 1) x (cols - 1) = ({n_rows}-1) x ({n_cols}-1) = {df}')
print(f'p-value (right tail) = {p:.4e}')
"""),
        md(r"""
### Check against `scipy.stats.chi2_contingency`

`scipy` does the same arithmetic and reads the p-value off the theoretical curve. It should agree with
what we built by hand.

**One gotcha for a 2×2 table:** by default `chi2_contingency` applies **Yates' continuity correction**, a
small tweak that *only* kicks in for 2×2 tables and gives a slightly different number. To match our plain
textbook formula, pass `correction=False`. (For tables bigger than 2×2 there's no correction, so they
match either way — we'll see that in Section 5.)
"""),
        code(r"""
chi2_sp, p_sp, df_sp, E_sp = stats.chi2_contingency(O, correction=False)

print('--- ours vs scipy.stats.chi2_contingency (Sex x Survived, 2x2) ---')
print(f'chi^2 :  ours {chi2:.4f}   scipy {chi2_sp:.4f}')
print(f'df    :  ours {df}         scipy {df_sp}')
print(f'p     :  ours {p:.4e}   scipy {p_sp:.4e}')
print('match?', np.isclose(chi2, chi2_sp) and df == df_sp and np.isclose(p, p_sp))

# What the DEFAULT (Yates-corrected) would have given on this 2x2 -- close but NOT equal:
chi2_yates = stats.chi2_contingency(O)[0]
print(f'\n(With the default correction=True, scipy reports chi^2 = {chi2_yates:.4f} -- the Yates tweak,')
print(' which is why we pass correction=False to reproduce the by-hand formula exactly.)')
"""),
        md(r"""
### A goodness-of-fit aside — `scipy.stats.chisquare`

Chi-square has a second flavour the core lesson covered: **goodness-of-fit (GOF)**, which checks a
*single* categorical variable against a *claimed* distribution. Same statistic $\sum (O-E)^2/E$, but now
$E$ comes from a stated set of proportions instead of an outer product. Suppose a historian claims the
class mix was **25% first, 20% second, 55% third**. Does the passenger list fit *that*? Here $E$ is just
`proportions * n`, and `scipy.stats.chisquare` reads off the p-value.
"""),
        code(r"""
class_counts = titanic['Pclass'].value_counts().sort_index().to_numpy()
n = class_counts.sum()
claimed = np.array([0.25, 0.20, 0.55])        # historian's stated class mix
expected_claim = claimed * n

chi2_gof_hand = np.sum((class_counts - expected_claim) ** 2 / expected_claim)
chi2_gof, p_gof = stats.chisquare(f_obs=class_counts, f_exp=expected_claim)

print('observed class counts:', class_counts.tolist(), '  (1st, 2nd, 3rd)')
print('expected (25/20/55)  :', expected_claim.round(1).tolist())
print(f'\nchi^2 by hand = {chi2_gof_hand:.3f}    scipy.stats.chisquare = {chi2_gof:.3f}')
print('match?', np.isclose(chi2_gof_hand, chi2_gof))
print(f'p-value = {p_gof:.3f}  -> large p: NO evidence the data departs from the 25/20/55 claim. A good fit.')
"""),
        md(r"""
## 4) The idiom — `np.outer` and a fully vectorized statistic

### Build the expected table with an outer product

That double loop in Section 3 was for clarity. The expected table is exactly an **outer product** of the
row and column totals, divided by the grand total — and `np.outer(a, b)` builds the whole grid in one
call: cell $(i, j)$ is `a[i] * b[j]`. Divide the result by the grand total and you have $E$, no loop in sight.
"""),
        code(r"""
E_vectorized = np.outer(row_totals, col_totals) / grand_total

print('E from the double loop:\n', np.round(E, 2))
print('\nE from np.outer       :\n', np.round(E_vectorized, 2))
print('\nidentical?', np.allclose(E, E_vectorized))
print('\nnp.outer(row_totals, col_totals) makes the (rows x cols) grid where cell (i, j)')
print('is row_totals[i] * col_totals[j] -- precisely the numerator of every expected count.')
"""),
        md(r"""
### The statistic with no loops at all

`(O - E)**2 / E` is **broadcasting** in action: the subtraction, the square and the division each run
across the *entire* table at once, producing a table of per-cell contributions. `.sum()` collapses that
table to the single $\chi^2$ number. Looking at the per-cell contributions is also a nice diagnostic — it
shows you *which* cells drove the surprise.
"""),
        code(r"""
contributions = (O.values - E_vectorized) ** 2 / E_vectorized     # one table of (O-E)^2/E values
chi2_vec = contributions.sum()                                    # add them all up

contrib_df = pd.DataFrame(contributions.round(1), index=O.index, columns=O.columns)
print('Per-cell contributions to chi^2:\n')
print(contrib_df)
print(f'\nchi^2 = sum of all cells = {chi2_vec:.4f}   (same as before: {np.isclose(chi2_vec, chi2)})')
print('\nNo loop over cells -- the whole computation is three array operations and a sum.')
"""),
        md(r"""
## 5) Refactor into a reusable function

You've now written the same steps — totals, outer-product expected table, vectorized sum, df, p-value — a
couple of times. The programmer's reflex is to **wrap repeated work in a function**: name it, give it an
input, `return` the result. Then "run a chi-square test of independence" is one call you can reuse on
*any* contingency table.
"""),
        code(r"""
def chi_square_independence(observed):
    '''Chi-square test of independence on a contingency table of counts.

    Pass a 2-D array-like (or a crosstab) of observed counts; get back a dict
    with the statistic, degrees of freedom, and p-value -- our own chi2_contingency.
    '''
    O = np.asarray(observed, dtype=float)
    row_totals = O.sum(axis=1)
    col_totals = O.sum(axis=0)
    grand_total = O.sum()
    E = np.outer(row_totals, col_totals) / grand_total      # expected under independence
    chi2 = np.sum((O - E) ** 2 / E)                          # vectorized statistic
    n_rows, n_cols = O.shape
    df = (n_rows - 1) * (n_cols - 1)
    p = stats.chi2.sf(chi2, df)
    return {'chi2': chi2, 'df': df, 'p': p}

# Reuse it on the Sex x Survived table and re-check against scipy:
result = chi_square_independence(O)
chi2_sp, p_sp, df_sp, _ = stats.chi2_contingency(O, correction=False)
print('Sex x Survived (2x2):')
print(f"  ours : chi^2={result['chi2']:.4f}  df={result['df']}  p={result['p']:.4e}")
print(f"  scipy: chi^2={chi2_sp:.4f}  df={df_sp}  p={p_sp:.4e}")
print('  match?', np.isclose(result['chi2'], chi2_sp) and result['df'] == df_sp and np.isclose(result['p'], p_sp))
"""),
        code(r"""
# The payoff of a function: reuse it unchanged on a BIGGER table -- ticket class x survival (3x2).
# For tables bigger than 2x2 there is no Yates correction, so scipy's DEFAULT matches us directly.
class_table = pd.crosstab(titanic['Pclass'], titanic['Survived'])
class_table.index = ['1st', '2nd', '3rd']
class_table.columns = ['died (0)', 'survived (1)']
print(class_table, '\n')

res_class = chi_square_independence(class_table)
chi2_c, p_c, df_c, _ = stats.chi2_contingency(class_table)   # default correction -- no effect on 3x2
print('Class x Survived (3x2):')
print(f"  ours : chi^2={res_class['chi2']:.4f}  df={res_class['df']}  p={res_class['p']:.4e}")
print(f"  scipy: chi^2={chi2_c:.4f}  df={df_c}  p={p_c:.4e}")
print('  match?', np.isclose(res_class['chi2'], chi2_c) and res_class['df'] == df_c and np.isclose(res_class['p'], p_c))
"""),
        md(r"""
A function is a contract: *give me a table of counts, I'll give you the statistic, df and p-value.* Once
it's correct, you stop thinking about outer products and degrees of freedom and start thinking in
**tables and verdicts** — the leap from "writing statistics" to "doing statistics in code." (And yes,
`scipy.stats.chi2_contingency` exists; the point was to *build* it so you know what it's doing — and so
the `correction=False` footnote stops being mysterious.)
"""),
        md(r"""
## 6) Now you code

Write the code yourself. Add a new cell (the **+** button in the toolbar) and try each — predict the
answer before you run it. Seed any randomness with `rng = np.random.default_rng(0)` so your runs repeat.

1. **The expected-counts $\ge 5$ check.** The chi-square curve is an approximation that wants every
   *expected* cell to be about 5 or more. Write a one-liner that prints the smallest expected count for
   the Sex × Survived table: `np.outer(row_totals, col_totals).min() / grand_total`. Is it comfortably
   above 5? (It is — about 121.)

2. **A Cramér's V effect size.** A huge $\chi^2$ says the link is *real*, not that it's *large*. Write
   `cramers_v(table)` returning $\sqrt{\chi^2 / (N\,(\min(r,c)-1))}$, reusing your
   `chi_square_independence` for the $\chi^2$. Test it on the Sex × Survived table (you should get about
   0.54 — a large effect).

3. **A child/adult column, tested against survival.** Build a two-category age variable and crosstab it:

       child = np.where(titanic['Age'] < 16, 'child', 'adult')
       tab = pd.crosstab(child, titanic['Survived'])
       print(tab)
       print(chi_square_independence(tab))

   Compare your dict to `stats.chi2_contingency(tab, correction=False)[:2]` (it's 2×2, so use
   `correction=False`). Is "children first" visible in the counts?

4. **See the Yates correction yourself.** On the Sex × Survived 2×2 table, print
   `stats.chi2_contingency(O, correction=False)[0]` and `stats.chi2_contingency(O)[0]` side by side.
   Confirm the first matches your `chi_square_independence(O)['chi2']` and the second (the default) is a
   little smaller. Now try it on the 3×2 class table — why are the two scipy numbers identical there?

## What you learned

**Coding**
- `pd.crosstab(var1, var2)` turns two categorical columns into a **contingency table** of counts; pull its
  raw numbers out with `.values` to do clean array math, and get the **marginals** with `.sum(axis=1)`
  (rows) and `.sum(axis=0)` (columns).
- **`np.outer(row_totals, col_totals)`** builds the whole expected-count grid in one call — cell $(i,j)$ is
  `row_totals[i] * col_totals[j]` — replacing a double loop over cells.
- **Broadcasting** lets `(O - E)**2 / E` run across the entire table at once; `.sum()` collapses it to the
  single statistic. The per-cell version is a handy diagnostic for *which* cells drove the surprise.
- **Refactor repeated work into a function** with a clear input and a `return` (here a dict), then reuse it
  unchanged on a bigger table.

**Statistics (now demystified)**
- The chi-square test of independence is: expected $E_{ij} = (\text{row}_i \times \text{col}_j)/N$, statistic
  $\chi^2 = \sum (O-E)^2/E$, $\text{df} = (r-1)(c-1)$, and the p-value is the right tail `stats.chi2.sf(chi2, df)`.
  You reproduced every `scipy.stats.chi2_contingency` number to the decimal.
- The expected table keeps the **same row and column totals** as the observed one — a built-in correctness check.
- For a **2×2** table, `scipy`'s default applies **Yates' continuity correction**; pass `correction=False` to
  match the textbook formula. Larger tables have no correction, so they match either way.

**Back to the core lesson** ([Lesson 17: Chi-square: goodness-of-fit & independence](17-chi-square.ipynb)) for the
*meaning* of these numbers, or read on through the course.
"""),
        md(nav_md(ID)),
    ]


if __name__ == "__main__":
    save_lesson(ID, cells())
