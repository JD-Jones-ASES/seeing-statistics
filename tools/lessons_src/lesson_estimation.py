r"""Lesson — Estimation & standard error (Ames housing as a known population). See course MANIFEST in lessonkit.py."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # tools/
from lessonkit import md, code, save_lesson, header_md, nav_md

ID = "estimation"

INTRO = r"""
You almost never get to measure *everyone*. Instead you measure a **sample** and use it to *guess* a
number about the whole population — the average house size, the share of voters, the typical commute.
That guess is an **estimate**, and the honest follow-up question is: **how far off is it likely to be?**

This lesson answers that with one of the most useful ideas in all of statistics: the **standard error**
— the typical distance between a sample estimate and the truth. We'll use a clever trick to *see* it:
treat a real dataset as a **known population** (so we actually know the true answer), draw thousands of
samples, and watch our estimates dance around the truth. Run each grey cell with **Shift+Enter**, then
tinker.
"""


def cells():
    return [
        md(header_md(ID, intro=INTRO)),
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
from scipy import stats
sl.use_course_style()

AMES_URL = 'https://www.openintro.org/data/csv/ames.csv'
ames = sl.load_csv('ames.csv', url=AMES_URL)

# Treat the WHOLE file as the population, so we get to KNOW the true answer.
population = ames['area'].dropna().to_numpy()    # living area (sq ft) of every home
MU    = population.mean()                          # the TRUE population mean  (mu)
SIGMA = population.std(ddof=0)                     # the TRUE population stdev (sigma)
N_POP = len(population)

print(f'Population: {N_POP:,} Ames homes (living area, sq ft)')
print(f'TRUE population mean   mu    = {MU:.1f} sq ft   <- the answer we will try to estimate')
print(f'TRUE population stdev  sigma = {SIGMA:.1f} sq ft')
"""),
        md(r"""
## 1) Parameter vs. statistic — the two kinds of number

The whole game of inference rests on telling apart two numbers that look similar but mean very
different things.

- A **parameter** is a number that describes the **whole population**. It is **fixed** but usually
  **unknown** — in real life you can't measure everyone. Greek letters: the population mean is $\mu$
  ("mu"), the population standard deviation is $\sigma$ ("sigma"), a population proportion is $p$.
- A **statistic** is a number computed from a **sample**. It is **known** (you have the data) but
  **random** — draw a different sample and you'd get a different value. Roman letters: the sample mean
  is $\bar{x}$ ("x-bar"), the sample standard deviation is $s$, a sample proportion is $\hat{p}$
  ("p-hat").

> **The mantra:** *parameters are fixed and unknown; statistics are random and known.* We use the
> statistic we *can* see to estimate the parameter we *can't*.

Our setup is special on purpose: because we declared the whole file the population, we know the
parameter $\mu \approx 1500$ sq ft. In a real study you would **never** know this — which is exactly why
the rest of the course exists. Here we know it so we can *check* whether our methods actually work.
"""),
        code(r"""
# A statistic is what we compute from a SAMPLE. Let's grab one sample of 50 homes.
rng = np.random.default_rng(11)          # change this seed to draw a different sample
n = 50
# Draw i.i.d. (replace=True): treat the file as an endless super-population, so
# sigma/sqrt(n) is the EXACT target for the standard error later on.
sample = rng.choice(population, size=n, replace=True)

xbar = sample.mean()                       # the STATISTIC: our estimate of mu
s    = sample.std(ddof=1)                   # sample standard deviation (note ddof=1)
print(f'Sample of n = {n} homes')
print(f'  sample mean  x-bar = {xbar:.1f} sq ft   (our estimate)')
print(f'  true mean    mu    = {MU:.1f} sq ft   (the target)')
print(f'  we missed by       = {xbar - MU:+.1f} sq ft')
"""),
        md(r"""
## 2) A statistic is an *estimator*

When we use the sample mean $\bar{x}$ to *stand in for* the unknown population mean $\mu$, we are using
it as an **estimator** — a recipe that turns sample data into a guess about a parameter. The single
number it produces (like the 1,?? sq ft above) is a **point estimate**: one best guess, a single point
on the number line.

| Parameter (unknown truth) | Estimator (the recipe) | Point estimate (one number) |
|---|---|---|
| population mean $\mu$ | sample mean $\bar{x}$ | e.g. 1,520 sq ft |
| population proportion $p$ | sample proportion $\hat{p}$ | e.g. 0.18 |
| population stdev $\sigma$ | sample stdev $s$ | e.g. 498 sq ft |

The same idea works for a **proportion**. Suppose we care about the share of "big" homes (say, over
2,000 sq ft). The parameter $p$ is the true share in the whole population; the estimator is the sample
proportion $\hat{p}$ = (big homes in the sample) / (sample size).
"""),
        code(r"""
# p-hat estimates p, exactly as x-bar estimates mu.
big = 2000                                  # call a home "big" if area > 2000 sq ft
p_true = (population > big).mean()           # the PARAMETER p (we know it here)
p_hat  = (sample > big).mean()               # the STATISTIC  p-hat (our estimate)

print(f'TRUE proportion of homes over {big} sq ft :  p     = {p_true:.3f}')
print(f'Sample estimate from our 50 homes        :  p-hat = {p_hat:.3f}')
print(f'\nBoth x-bar->mu and p-hat->p are point estimates: one number guessing a parameter.')
"""),
        md(r"""
## 3) Bias vs. precision — the dartboard

A single estimate is one dart thrown at a board whose bullseye is the true parameter $\mu$. To judge an
estimator we throw *many* darts (take many samples) and ask two separate questions:

- **Bias (accuracy):** do the darts *center* on the bullseye, or are they systematically pulled to one
  side? An estimator is **unbiased** if, averaged over all possible samples, it lands exactly on the
  truth — its long-run average equals the parameter. The sample mean $\bar{x}$ is unbiased for $\mu$.
- **Precision (spread):** how *tightly* do the darts cluster? A precise estimator gives nearly the same
  answer every time; an imprecise one is all over the board. We measure this spread with the **standard
  error** (next section).

These are independent. You can be:

| | tight cluster (precise) | wide cluster (imprecise) |
|---|---|---|
| **centered on truth (unbiased)** | best case: accurate *and* reliable | right on average, but any one shot is shaky |
| **off-center (biased)** | reliably *wrong* (the dangerous case!) | wrong *and* shaky |

Let's throw the darts. We'll draw **3,000 samples**, compute $\bar{x}$ from each, and see where they land.
"""),
        code(r"""
n = 50
n_samples = 3000
xbars = np.array([rng.choice(population, size=n, replace=True).mean()
                  for _ in range(n_samples)])

print(f'Drew {n_samples:,} samples of n = {n}.')
print(f'Average of all the sample means : {xbars.mean():.1f} sq ft')
print(f'True population mean mu          : {MU:.1f} sq ft')
print(f'Difference (the bias)           : {xbars.mean() - MU:+.2f} sq ft   <- essentially 0: UNBIASED')
"""),
        code(r"""
# The dartboard, drawn as a histogram of where x-bar lands.
fig, ax = plt.subplots()
ax.hist(xbars, bins=40, color='#4c72b0', edgecolor='white')
ax.axvline(MU, color='#d62728', lw=2.5, ls='--', label=f'true mean mu = {MU:.0f}')
ax.axvline(xbars.mean(), color='black', lw=2, label=f'avg of estimates = {xbars.mean():.0f}')
ax.set_xlabel('sample mean x-bar (sq ft)'); ax.set_ylabel('number of samples')
ax.set_title(f'{n_samples:,} estimates of the mean — centered on the truth (unbiased)')
ax.legend(); plt.show()
"""),
        md(r"""
The cloud of estimates piles up **right on top of** the true mean (the black "average of estimates"
line sits essentially on the red "truth" line). That's what **unbiased** looks like: no systematic pull
to either side. The *width* of that cloud — how far a typical dart lands from center — is the thing we
quantify next.
"""),
        md(r"""
## 4) The standard error — the typical miss

The **standard error (SE)** of an estimator is just the **standard deviation of its sampling
distribution** — that cloud of estimates we just drew. In plain words:

> **The standard error is the typical distance between one sample's estimate and the true parameter.**

It is the number that turns "my estimate is off by *some* amount" into "my estimate is off by *about
this much*." For the sample mean, theory gives a beautifully simple formula:

$$\text{SE}(\bar{x}) \;=\; \frac{\sigma}{\sqrt{n}}$$

But there's a catch: in real life we don't know $\sigma$ (that's a population parameter!). So we
**estimate** the standard error by swapping in the sample standard deviation $s$:

$$\widehat{\text{SE}}(\bar{x}) \;=\; \frac{s}{\sqrt{n}}$$

This $s/\sqrt{n}$ is what calculators and textbooks call "the standard error." Let's prove it's right
the only way that really convinces anyone — by **simulation**. We already have 3,000 sample means; their
*actual* spread should match $\sigma/\sqrt{n}$.
"""),
        code(r"""
empirical_SE = xbars.std(ddof=0)            # the ACTUAL spread of our 3,000 estimates
formula_SE   = SIGMA / np.sqrt(n)           # the textbook prediction sigma/sqrt(n)

print(f'Empirical SE  (spread of the 3,000 x-bars) = {empirical_SE:.2f} sq ft')
print(f'Formula   SE  (sigma / sqrt(n))            = {formula_SE:.2f} sq ft')
print(f'They match to within {abs(empirical_SE - formula_SE):.2f} sq ft  ->  the formula is REAL.')
"""),
        md(r"""
The simulated spread and the formula $\sigma/\sqrt{n}$ agree to a hair. The standard-error formula isn't
a magic incantation — it is literally *predicting how spread out our estimates turned out to be*, and it
nails it.

Now the **practical** version: in a real study you have **one** sample, not 3,000. You don't know
$\sigma$, so you use $s/\sqrt{n}$. Does that single-sample estimate of the SE land near the truth?
"""),
        code(r"""
# In real life you get ONE sample and must estimate the SE from it alone.
one_sample = rng.choice(population, size=n, replace=True)
s_one  = one_sample.std(ddof=1)
SE_hat = s_one / np.sqrt(n)                  # estimated standard error, from one sample

print(f'From a single sample of n = {n}:')
print(f'  sample mean  x-bar       = {one_sample.mean():.1f} sq ft')
print(f'  estimated SE = s/sqrt(n) = {SE_hat:.2f} sq ft')
print(f'  (true SE = sigma/sqrt(n) = {formula_SE:.2f} sq ft, so our one-sample guess is close)')
print(f'\nReport it as:  mu is about {one_sample.mean():.0f} sq ft, give or take ~{SE_hat:.0f} sq ft.')
"""),
        md(r"""
### Standard deviation vs. standard error — don't mix them up

This trips up almost everyone, so let's nail it:

- **Standard deviation ($s$ or $\sigma$)** describes the spread of the **individual data points** — how
  much one *house* differs from another. It does **not** shrink as you collect more data; houses are as
  varied as they ever were.
- **Standard error ($s/\sqrt{n}$)** describes the spread of the **estimate** — how much the *sample mean*
  would differ from sample to sample. It **shrinks** as $n$ grows, because averaging more homes steadies
  the average.

Same units (sq ft), completely different jobs. The SD answers "how varied are the homes?"; the SE
answers "how trustworthy is my average?"
"""),
        code(r"""
print(f'Spread of individual HOMES (sample SD, s)      = {s_one:.1f} sq ft   <- big; homes vary a lot')
print(f'Spread of the AVERAGE      (standard error)    = {SE_hat:.1f} sq ft   <- small; averaging steadies it')
print(f'\nThe SE is smaller than s by a factor of sqrt(n) = sqrt({n}) = {np.sqrt(n):.1f}.')
"""),
        md(r"""
## 5) The $\sqrt{n}$ law — quadruple the sample, halve the error

Look hard at $\text{SE} = \sigma/\sqrt{n}$. The sample size sits under a **square root**. That single
detail governs the entire economics of data collection:

- To make your estimate **twice as precise** (cut the SE in half), you don't need *twice* the data — you
  need **four times** as much ($\sqrt{4} = 2$).
- For **10 times** the precision, you need **100 times** the data ($\sqrt{100} = 10$).

This is the law of *diminishing returns* in sampling: each extra data point helps, but later ones help
less than earlier ones. Let's watch the SE shrink as $n$ climbs — and confirm the simulated spread tracks
the $1/\sqrt{n}$ curve.
"""),
        code(r"""
sizes = [10, 25, 50, 100, 200, 400, 800]
rows = []
for nn in sizes:
    means = np.array([rng.choice(population, size=nn, replace=True).mean() for _ in range(2000)])
    rows.append({'n': nn,
                 'empirical_SE': means.std(ddof=0),
                 'formula_SE': SIGMA / np.sqrt(nn)})
se_table = pd.DataFrame(rows)
print(se_table.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

print(f'\nNotice: going from n=50 to n=200 (4x the data) roughly HALVES the SE:')
se50  = se_table.loc[se_table['n'] == 50,  'formula_SE'].iloc[0]
se200 = se_table.loc[se_table['n'] == 200, 'formula_SE'].iloc[0]
print(f'  SE at n=50  = {se50:.1f} sq ft')
print(f'  SE at n=200 = {se200:.1f} sq ft   (about half — quadrupling n halved the error)')
"""),
        code(r"""
# Picture the sqrt(n) law: dots = simulation, smooth curve = sigma/sqrt(n).
nn_grid = np.linspace(10, 800, 200)
fig, ax = plt.subplots()
ax.plot(nn_grid, SIGMA / np.sqrt(nn_grid), color='#d62728', lw=2.5,
        label=r'formula  SE = sigma / sqrt(n)')
ax.scatter(se_table['n'], se_table['empirical_SE'], s=60, color='#4c72b0', zorder=5,
           label='simulated spread of x-bar')
ax.set_xlabel('sample size n'); ax.set_ylabel('standard error of the mean (sq ft)')
ax.set_title('The sqrt(n) law: bigger samples shrink the error (with diminishing returns)')
ax.legend(); plt.show()
"""),
        md(r"""
The simulated dots fall right on the $\sigma/\sqrt{n}$ curve, and the curve flattens out: the drop from
$n=10$ to $n=100$ is dramatic, but from $n=400$ to $n=800$ it's barely visible. That flattening is the
$\sqrt{n}$ law telling you when more data stops being worth the cost.
"""),
        md(r"""
## 6) From standard error to the margin of error (a bridge)

The standard error is the engine; the **margin of error** is what gets *reported*. A point estimate
alone is overconfident — it pretends we hit the bullseye exactly. Instead we surround it with a cushion:

$$\text{estimate} \;\pm\; (\text{a multiplier}) \times \text{SE}.$$

That "$\text{multiplier} \times \text{SE}$" piece is the **margin of error**. For roughly 95%
confidence the multiplier is about **2** (it comes from the bell curve — see the normal-distribution and
Central Limit Theorem lessons), so a rough 95% interval is

$$\bar{x} \;\pm\; 2\,\frac{s}{\sqrt{n}}.$$

Stretching the point estimate into a *range* like this gives the **confidence interval**. Let's build a
quick one — then leave the full story (where the "2" really comes from, what "95% confidence" honestly
means, and a 100-interval coverage demo) to **the confidence-interval flagship**, the very next stop.
"""),
        code(r"""
estimate = one_sample.mean()
moe = 2 * SE_hat                            # rough 95% margin of error  (~2 * SE)
lo, hi = estimate - moe, estimate + moe

print(f'Point estimate  : x-bar = {estimate:.0f} sq ft')
print(f'Margin of error : +/- {moe:.0f} sq ft   ( ~2 * SE )')
print(f'Rough 95% interval: ({lo:.0f}, {hi:.0f}) sq ft')
print(f'True mean mu      : {MU:.1f} sq ft  ->  {"inside" if lo <= MU <= hi else "outside"} our interval')
print('\nNext lesson: what "95% confidence" really means, and why the multiplier is ~2.')
"""),
        md(r"""
## Now you try

Predict what will happen *before* you run each change:

1. **Shrink the sample.** In the section-3/4 cells, set `n = 10` and re-run. Does the histogram of
   estimates get **wider** or narrower? Does the empirical SE still match $\sigma/\sqrt{n}$? (It should —
   the formula works at every $n$.)

2. **Test the $\sqrt{n}$ law by hand.** The SE at `n = 50` is printed above. Predict the SE at
   `n = 800` *before looking* — divide the `n=50` value by $\sqrt{800/50} = \sqrt{16} = 4$. Then check it
   against the table in section 5.

3. **Estimate a proportion instead of a mean.** In a new cell, draw a sample and compute `p-hat` for
   "homes over 2000 sq ft" the way section 2 does, but do it **2,000 times** and take the standard
   deviation of all the `p-hat` values. That spread is the standard error of a *proportion*. (Bonus: it
   should be close to $\sqrt{p(1-p)/n}$ — the proportion's own SE formula.)

4. **Change the seed.** Swap `np.random.default_rng(11)` for any other number. Your single point estimate
   moves, but does the *standard error* (the typical miss) change much? Why or why not?

## What you learned

- A **parameter** describes the whole population (fixed, unknown, Greek: $\mu$, $\sigma$, $p$); a
  **statistic** is computed from a sample (random, known, Roman: $\bar{x}$, $s$, $\hat{p}$).
- A statistic used to guess a parameter is an **estimator**; the single number it gives is a **point
  estimate** ($\bar{x}$ estimates $\mu$; $\hat{p}$ estimates $p$).
- **Bias** is whether estimates *center* on the truth (accuracy); **precision** is how *tightly* they
  cluster (spread). The sample mean is **unbiased** — we saw 3,000 estimates pile up right on $\mu$.
- The **standard error** is the typical distance from estimate to truth — the standard deviation of the
  sampling distribution. For a mean it is $\sigma/\sqrt{n}$, estimated in practice by $s/\sqrt{n}$, and
  simulation confirmed the formula exactly.
- **Standard deviation** measures the spread of individual data points and doesn't shrink with $n$;
  **standard error** measures the spread of the *estimate* and shrinks like $1/\sqrt{n}$.
- The **$\sqrt{n}$ law**: quadruple the sample to halve the error — real but with diminishing returns.
- **Margin of error** $\approx 2 \times \text{SE}$ stretches a point estimate into a range — the bridge
  to the confidence-interval flagship coming next.
"""),
        md(nav_md(ID)),
    ]


if __name__ == "__main__":
    save_lesson(ID, cells())
