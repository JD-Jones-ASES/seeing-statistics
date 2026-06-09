r"""Lesson 19 (code variant) — "The code behind it": build multiple regression
from scratch with linear algebra, on the SAME Ames housing predictors (price ~
area + quality + year_built) as the core lesson. Assemble the design matrix with
a leading column of ones, solve OLS with np.linalg.lstsq, compute R^2 and
adjusted R^2, verify against statsmodels, and watch a slope change when a
predictor is added. Same skeleton as lesson_01_code.py. See MANIFEST in
lessonkit.py."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # tools/
from lessonkit import md, code, save_lesson, header_md, nav_md

ID = "multiple-regression-code"

INTRO = r"""
**In Lesson 19 you fit several predictors at once with one tidy line** — `smf.ols('price ~ area +
quality + year_built', data=ames).fit()` — and read the coefficients off a table. It worked like a
magic word. This companion **opens the hood**: you'll build that same fit *yourself* out of a matrix and
a single call to a linear-algebra solver, so a "multiple regression" stops being a formula string and
becomes a few lines of NumPy you could write from memory.

Same Ames homes, same predictors, same final coefficients — but the focus shifts from *what the
coefficients mean* to **how you compute them**. We'll go in the order a programmer actually works:

1. **The goal** — what we're building.
2. **The data in code** — load it, then assemble the **design matrix** `X` and response `y`.
3. **Build it step by step** — solve for the coefficients with linear algebra, and check them against statsmodels.
4. **The idiom** — matrix multiplication with `@`, and `np.linalg.lstsq`.
5. **Refactor into a function** — wrap it all up as your own reusable `ols_fit()`.
6. **Now you code** — write-it-yourself exercises.
7. **What you learned** — the coding *and* the statistics.

You don't need Lesson 19 fresh in mind, but it helps. Run each grey cell with **Shift + Enter**, top to bottom.
"""


def cells():
    return [
        md(header_md(ID, intro=INTRO)),
        md(r"""
## 1) The goal

By the end of this notebook you'll have written a function that takes a table of predictors and a column
of outcomes and returns the regression coefficients — the same numbers `statsmodels` hands you for free —
plus the model's $R^2$ and **adjusted** $R^2$. Nothing here needs new statistics; everything is the linear
algebra *underneath* the one-line `smf.ols(...)` call.

The whole of multiple regression is one compact equation. We stack the predictors into a matrix $X$ (with
a leading column of **ones** for the intercept) and the outcomes into a vector $y$, then find the
coefficient vector $\beta$ that makes $X\beta$ as close to $y$ as possible:

$$\hat{y} = X\beta, \qquad \beta = \arg\min_\beta \; \lVert y - X\beta \rVert^2.$$

The classic "normal equations" solve it as $\beta = (X^\top X)^{-1} X^\top y$, but we'll compute $\beta$
with `np.linalg.lstsq`, which does the same least-squares fit **more stably** (no explicit matrix inverse).

The tools we'll use, and what each is for:

| Tool | What it is | What we use it for |
|---|---|---|
| **pandas** | tables (`DataFrame`) and columns (`Series`) | loading the CSV, picking the predictor columns |
| **NumPy** | fast math on whole arrays and matrices | the design matrix, `@`, and `np.linalg.lstsq` |
| **statsmodels** | a statistics library | the trusted answer we check ourselves against |
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
import statsmodels.api as sm
sl.use_course_style()

AMES_URL = 'https://www.openintro.org/data/csv/ames.csv'
ames = sl.load_csv('ames.csv', url=AMES_URL)

# Same clean names and same predictors as the core lesson.
ames = ames.rename(columns={'Overall.Qual': 'quality', 'Year.Built': 'year_built'})
ames = ames.dropna(subset=['price', 'area', 'quality', 'year_built'])

print(f'Loaded a {type(ames).__name__} with shape {ames.shape}  (rows, columns)')
print(f'{len(ames):,} homes (Ames, Iowa) after dropping rows missing any predictor or the price.')
ames[['price', 'area', 'quality', 'year_built']].head()
"""),
        md(r"""
## 2) The data in code — build the design matrix

A regression solver doesn't want a tidy `DataFrame`; it wants two plain NumPy arrays:

- **`y`** — the response, a 1-D array of the values we're predicting (`price`).
- **`X`** — the **design matrix**, one row per home and one column per predictor, *plus a leading column
  of ones*. That column of ones is the secret of the **intercept**: multiplying it by a coefficient lets
  the model add a constant baseline $b_0$, exactly like the `Intercept` row in the core lesson's table.

Two habits worth keeping when you hand-build a design matrix:

- **Drop the NaNs across the response and every predictor together** (we already did, in the setup) so
  every row of `X` lines up with a value of `y`.
- **Convert pandas to NumPy with `.to_numpy()`** before the linear algebra, so nothing silently re-aligns
  on a pandas index. We force `float` too — least squares is arithmetic, not integers.
"""),
        code(r"""
predictors = ['area', 'quality', 'year_built']        # the SAME three predictors as the core lesson

y = ames['price'].to_numpy(dtype=float)               # the response, a 1-D array
n = len(y)

ones = np.ones(n)                                     # the intercept column: a 1 for every home
cols = [ames[c].to_numpy(dtype=float) for c in predictors]
X = np.column_stack([ones] + cols)                    # stack ones + the three predictors, side by side

col_names = ['intercept'] + predictors                # to label each coefficient later
print('X shape:', X.shape, ' (rows = homes, columns = intercept + 3 predictors)')
print('y shape:', y.shape)
print('\ncolumns of X:', col_names)
print('\nfirst 3 rows of X (note the leading 1s = the intercept column):')
print(np.round(X[:3], 1))
"""),
        md(r"""
## 3) Build it step by step — solve OLS (ordinary least squares) with linear algebra

Now the heart of it. We hand `X` and `y` to `np.linalg.lstsq`, which returns the coefficient vector
$\beta$ that minimizes the sum of squared misses $\sum (y_i - \hat{y}_i)^2$ — the very same least-squares
criterion the core lesson's `smf.ols(...)` uses. (`rcond=None` just opts into the modern default; the
`[0]` plucks the coefficients out of the tuple it returns.)

Each entry of $\beta$ pairs with a column of $X$: the first with the ones-column (the intercept), then one
per predictor. Once you've built it, the regression table is no longer a mystery box.
"""),
        code(r"""
beta = np.linalg.lstsq(X, y, rcond=None)[0]     # the least-squares coefficients, all at once

print('Coefficients (our own fit):')
for name, b in zip(col_names, beta):
    print(f'   {name:<12} {b:>16,.4f}')

print(f'\nReading it back as an equation:')
print(f'   price = {beta[0]:,.0f} + {beta[1]:.2f}*area + {beta[2]:,.0f}*quality + {beta[3]:.2f}*year_built')
print(f'\n-> holding quality and age fixed, an extra square foot is worth about {beta[1]:.0f} dollars.')
"""),
        md(r"""
### Fit quality: residuals, $R^2$, and adjusted $R^2$

How well does the flat sheet fit? First the **predictions** $\hat{y} = X\beta$ — one matrix multiply gives
all 2,930 fitted prices at once. The **residuals** are what's left over, $y - \hat{y}$.

$R^2$ is the share of price's variation the model explains: 1 minus the ratio of *leftover* spread to
*total* spread. **Adjusted** $R^2$ then charges a penalty for each predictor (so junk columns can't inflate
it), with $p$ = number of predictors (not counting the intercept):

$$R^2 = 1 - \frac{\sum (y_i-\hat{y}_i)^2}{\sum (y_i-\bar{y})^2}, \qquad
R^2_{\text{adj}} = 1 - (1-R^2)\,\frac{n-1}{\,n-p-1\,}.$$
"""),
        code(r"""
yhat = X @ beta                       # predictions: one matrix multiply, all homes at once
resid = y - yhat                      # residuals: what the model missed on each home

ss_res = (resid ** 2).sum()           # leftover (unexplained) variation
ss_tot = ((y - y.mean()) ** 2).sum()  # total variation in price
r2 = 1 - ss_res / ss_tot

p = len(predictors)                   # number of predictors, EXCLUDING the intercept
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f'n = {n:,} homes, p = {p} predictors')
print(f'R-squared          : {r2:.6f}   ->  the model explains {r2*100:.1f}% of the variation in price')
print(f'adjusted R-squared : {adj_r2:.6f}')
"""),
        md(r"""
### Verify against statsmodels — same `X`, same answer

The real test: do our hand-built numbers match the trusted library? We fit `sm.OLS(y, X)` — note we pass
**the same `X` that already contains the ones-column**, so we do *not* call `sm.add_constant` (that would
add a *second* intercept column and break the match). Then we compare coefficients, $R^2$, and adjusted
$R^2$ with `np.allclose`.
"""),
        code(r"""
model = sm.OLS(y, X).fit()            # X already has the intercept column -> do NOT add_constant

print('Our beta vs. statsmodels params:')
for name, mine, theirs in zip(col_names, beta, model.params):
    print(f'   {name:<12} ours={mine:>16,.4f}   statsmodels={theirs:>16,.4f}')

print('\nmatch? (coefficients) :', np.allclose(beta, model.params))
print('match? (R-squared)    :', np.isclose(r2, model.rsquared))
print('match? (adjusted R^2) :', np.isclose(adj_r2, model.rsquared_adj))
print(f'\n   our R^2 = {r2:.6f}   statsmodels = {model.rsquared:.6f}')
print(f'   our adj = {adj_r2:.6f}   statsmodels = {model.rsquared_adj:.6f}')
"""),
        md(r"""
### "Holding the others constant" — watch a slope change

The core lesson's punchline was that area's value-per-square-foot **drops by roughly half** the moment the
model also sees quality — that's **confounding** (bigger homes tend to be higher quality, so the simple
slope was secretly paid for both). We can reproduce that with our own machinery: fit the **simple** model
`price ~ area` (a design matrix with just ones and area), then compare area's slope to the **multiple**
model from above.
"""),
        code(r"""
# Simple regression: intercept + area only.
X_simple = np.column_stack([np.ones(n), ames['area'].to_numpy(dtype=float)])
beta_simple = np.linalg.lstsq(X_simple, y, rcond=None)[0]

slope_alone = beta_simple[1]          # area's slope with NOTHING else in the model
slope_adjusted = beta[1]             # area's slope holding quality + year_built fixed (from section 3)

print('Slope on AREA (dollars per extra square foot):')
print(f'   alone (price ~ area)                       : {slope_alone:7.2f}')
print(f'   adjusted (price ~ area + quality + year)   : {slope_adjusted:7.2f}   <- roughly HALVED')
print(f'\nThat drop from ~{slope_alone:.0f} to ~{slope_adjusted:.0f} is confounding: the simple slope was crediting')
print('area for the quality (and age) that tends to ride along with bigger homes. Once quality and')
print('year_built claim their own share, what is left for area is its value among comparable homes.')
"""),
        md(r"""
## 4) The idiom — matrix operations and `np.linalg.lstsq`

### `@` is matrix multiplication

Three pieces of NumPy did all the work, and they're the idioms to remember:

- **Build the design matrix** with `np.column_stack([...])` — glue 1-D arrays together as columns. The
  leading `np.ones(n)` *is* the intercept.
- **Solve** with `np.linalg.lstsq(X, y, rcond=None)[0]` — the numerically stable least-squares fit.
- **Predict** with the `@` operator: `X @ beta` is matrix multiplication, computing all $n$ predictions in
  one shot. No loop over homes — same vectorized spirit as `mass.mean()` back in Lesson 1, scaled up to
  matrices.

Below we confirm `@` really is just "row times coefficients, summed": the first prediction by hand equals
the first entry of `X @ beta`.
"""),
        code(r"""
# What X @ beta does for ONE home: multiply that row by beta, element-by-element, and sum.
row0 = X[0]                                   # [1, area, quality, year_built] for the first home
pred0_by_hand = (row0 * beta).sum()           # = 1*b0 + area*b1 + quality*b2 + year*b3
pred0_matrix = (X @ beta)[0]                  # the same number, straight from the matrix multiply

print(f'first home, prediction by hand   : {pred0_by_hand:,.2f}')
print(f'first home, (X @ beta)[0]        : {pred0_matrix:,.2f}')
print('match?', np.isclose(pred0_by_hand, pred0_matrix))

# And the normal-equations FORM, for the curious (X.T is X "transposed" -- its rows and columns
# swapped; the @ symbol is matrix multiplication): beta = (X^T X)^{-1} X^T y.
# We computed beta with lstsq (more stable), but the explicit formula lands in the same place:
beta_normal = np.linalg.inv(X.T @ X) @ X.T @ y
print('\nlstsq vs. the explicit (X^T X)^{-1} X^T y formula match?', np.allclose(beta, beta_normal))
print('(lstsq is preferred in practice — it avoids inverting X^T X, which can be numerically touchy.)')
"""),
        md(r"""
## 5) Refactor into a reusable function

You've now assembled a design matrix and called `lstsq` a couple of times. The programmer's reflex is to
**wrap the repeated work in a function**: name it, give it inputs, `return` the results. Then "fit a
regression" is one call you can reuse on *any* set of predictors — three columns, one column, a different
outcome entirely.

Our `ols_fit` takes the `DataFrame`, the name of the response column, and a list of predictor names; it
builds `X` (with the ones-column), solves for `beta`, and returns the coefficients alongside $R^2$ and
adjusted $R^2$.
"""),
        code(r"""
def ols_fit(df, response, predictors):
    '''Fit ordinary least squares by linear algebra.

    Returns (coefficients dict, R^2, adjusted R^2). The design matrix gets a
    leading column of ones for the intercept, so don't add one yourself.
    '''
    data = df.dropna(subset=[response] + predictors)      # align response + all predictors
    y = data[response].to_numpy(dtype=float)
    n = len(y)
    X = np.column_stack([np.ones(n)] + [data[c].to_numpy(dtype=float) for c in predictors])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    p = len(predictors)
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    coefs = dict(zip(['intercept'] + predictors, beta))
    return coefs, r2, adj


# Reuse it on the full model, and re-check against statsmodels in one go.
coefs, r2_fn, adj_fn = ols_fit(ames, 'price', ['area', 'quality', 'year_built'])
for name, b in coefs.items():
    print(f'   {name:<12} {b:>16,.4f}')
print(f'   R^2 = {r2_fn:.6f}   adjusted R^2 = {adj_fn:.6f}')

check = sm.OLS(ames['price'].to_numpy(dtype=float),
               sm.add_constant(ames[['area', 'quality', 'year_built']].to_numpy(dtype=float))).fit()
print('\nmatch statsmodels?', np.allclose(list(coefs.values()), check.params),
      'and', np.isclose(r2_fn, check.rsquared), 'and', np.isclose(adj_fn, check.rsquared_adj))
"""),
        md(r"""
A function is a contract: *give me a table, a response, and some predictors; I'll give you the
coefficients and the fit.* Once it's correct, you stop thinking about design matrices and start thinking
in **predictors and adjusted effects** — exactly the leap from "writing regressions" to "doing regression
in code." (And yes, `smf.ols` exists; the point was to *build* it so you know what it's doing.)
"""),
        md(r"""
## 6) Now you code

Write the code yourself. Add a new cell (the **+** button in the toolbar) and try each — predict the answer
before you run it.

1. **Fit with two predictors.** Call `ols_fit(ames, 'price', ['area', 'quality'])` and read off area's
   slope. It should land *between* the simple-model slope (~112) and the three-predictor slope (~63) — each
   predictor you add re-adjusts the rest. Confirm the coefficients against
   `sm.OLS(y, sm.add_constant(ames[['area','quality']].to_numpy(float))).fit().params`.

2. **Does quality shrink too?** Use `ols_fit` to fit `price ~ quality` alone, then `price ~ area + quality`,
   and compare quality's coefficient across the two. Does quality's slope *also* drop once area is in the
   model? What does that tell you about which predictor "owns" more of the shared credit?

3. **Build the prediction yourself.** For the full model, take any single home's row of `X` and compute its
   predicted price as `(row * beta).sum()`. Check it equals the matching entry of `X @ beta`, and compare
   the prediction to that home's actual `price` (the difference is its residual).

4. **Adjusted $R^2$ vs. junk.** Add a column of pure noise — `rng = np.random.default_rng(0)` then
   `ames2 = ames.assign(noise=rng.normal(size=len(ames)))` — and call
   `ols_fit(ames2, 'price', ['area','quality','year_built','noise'])`. Plain $R^2$ will tick *up* (it always
   does), but does **adjusted** $R^2$ go *down*? (It should — the penalty sees through the useless column.)

## What you learned

**Coding**
- A **design matrix** `X` is how a regression sees your data: one row per observation, one column per
  predictor, plus a leading **column of ones** that becomes the intercept.
- Convert pandas to NumPy with `.to_numpy()` (and `dtype=float`) before linear algebra, and `dropna` across
  the response *and* every predictor together so the rows line up.
- **`np.linalg.lstsq(X, y, rcond=None)[0]`** solves least squares in one call — numerically steadier than
  inverting $X^\top X$ yourself, though the explicit `(X^T X)^{-1} X^T y` form lands in the same place.
- **`@` is matrix multiplication**: `X @ beta` computes every prediction at once — vectorization, scaled up
  from arrays to matrices.
- **Refactor repeated work into a function** (`ols_fit`) with clear inputs and a `return`, then reuse it.

**Statistics (now demystified)**
- Multiple regression is just $\hat{y} = X\beta$ solved by least squares; each coefficient is the
  **adjusted** effect of its predictor, holding the others fixed. You reproduced every statsmodels
  coefficient to the decimal.
- $R^2 = 1 - SS_{\text{res}}/SS_{\text{tot}}$, and **adjusted** $R^2$ penalizes extra predictors with the
  $(n-1)/(n-p-1)$ factor — both matched the library.
- A slope is **not** a fixed property of a predictor: area's value-per-square-foot dropped from ~112 to ~63
  once quality and age entered the model. That change *is* **confounding**, the regression face of
  *correlation is not causation*.

**Back to the core lesson** ([Lesson 19: Multiple regression (intro)](19-multiple-regression.ipynb)) for the
*meaning* of these numbers, or read on through the course.
"""),
        md(nav_md(ID)),
    ]


if __name__ == "__main__":
    save_lesson(ID, cells())
