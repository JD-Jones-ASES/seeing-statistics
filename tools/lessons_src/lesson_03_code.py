r"""Lesson 03 (code variant) — "The code behind it": build marginal, joint and
conditional probabilities straight from a pd.crosstab of the same Titanic data the
core lesson uses, verify every number against a direct boolean-mask computation,
do one step of Bayes by flipping the conditioning, and refactor into a reusable
cond_prob() function. Mirrors lesson_01_code.py's skeleton. See MANIFEST in lessonkit.py."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # tools/
from lessonkit import md, code, save_lesson, header_md, nav_md

ID = "03-code"

INTRO = r"""
**In Lesson 5 you counted probabilities off the Titanic** — `P(survived)`, `P(survived | female)`, the
multiplication rule, a Bayes flip. The numbers came out of `.mean()` calls and a `pd.crosstab` that felt a
little magical. This companion **opens the hood**: you'll build every one of those probabilities *yourself*,
straight from a table of counts, so "conditional probability" stops being a phrase and becomes a division you
could write from memory.

Same 887 passengers, same final numbers — but the focus shifts from *what the probability means* to **how you
compute it in Python**. We'll go in the order a programmer actually works:

1. **The goal** — what we're building.
2. **The data in code** — load it, look at its shape, make the boolean masks everything rests on.
3. **Build it step by step** — pull marginal, joint and conditional probabilities out of a count table by hand, and check each against a direct mask with `np.isclose`.
4. **The idiom** — the mean of a 0/1 column *is* a probability, and `crosstab(normalize=...)` reproduces the joint/conditional tables for free.
5. **Refactor into a function** — wrap it all up as your own reusable `cond_prob()`.
6. **Now you code** — predict-then-run exercises.
7. **What you learned** — the coding *and* the statistics.

You don't need Lesson 5 fresh in mind, but it helps. Run each grey cell with **Shift + Enter**, top to bottom.
"""


def cells():
    return [
        md(header_md(ID, intro=INTRO)),
        md(r"""
## 1) The goal

By the end of this notebook you'll have pulled three different kinds of probability — **marginal**, **joint**
and **conditional** — out of a single grid of counts, and verified each one against a direct calculation on
the raw rows. Then you'll wrap "conditional probability" into your own one-line function. Nothing here needs
new statistics; everything is the Python *underneath* the one-liners from Lesson 5.

The tools we'll use, and what each is for:

| Tool | What it is | What we use it for |
|---|---|---|
| **pandas** | tables (`DataFrame`) and columns (`Series`) | loading the CSV, masks, `crosstab`, `groupby` |
| **NumPy** | fast math on whole arrays at once | `np.isclose` to check our hand-built numbers |

Two ideas carry the whole lesson:

- A **boolean mask** is a column of `True`/`False` (e.g. `df['Sex'] == 'female'`). Python counts `True` as 1,
  so the **mean of a mask is the fraction that are True** — a probability.
- A **contingency table** (built by `pd.crosstab`) is just a grid of counts. Every probability we want is one
  number in that grid divided by another (a grand total, a row total, or a column total).
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
sl.use_course_style()

TITANIC_URL = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
titanic = sl.load_csv('titanic.csv', url=TITANIC_URL)
print(f'Loaded a {type(titanic).__name__} with shape {titanic.shape}  (rows, columns)')
"""),
        md(r"""
## 2) The data in code

Before any probability, a programmer *interrogates the object*: how big is it, what are the columns, what do
the first rows look like, how are the key columns coded? Same habits as always:

- `.shape` → `(rows, columns)`.
- `.columns` → the column names.
- `.head()` → the first few rows, to eyeball it.
"""),
        code(r"""
print('shape  :', titanic.shape)
print('columns:', list(titanic.columns))
titanic.head()
"""),
        md(r"""
### The two columns we need, and how they're coded

We only need two columns: **`Survived`** and **`Sex`**. Check exactly how each is stored — the whole lesson
depends on it.

- `Survived` is already **0 / 1** (0 = died, 1 = survived). A 0/1 column is gold: its `.mean()` is the
  survival *rate* directly, because the 1s are the only thing that add up.
- `Sex` is text, `'female'` / `'male'`.

This data has no missing values in these two columns, so there's nothing to drop — we read `raw/` and never
edit it.
"""),
        code(r"""
print('Survived values:', sorted(titanic['Survived'].unique()))
print('Sex values     :', sorted(titanic['Sex'].unique()))
print('missing in Survived/Sex:', titanic[['Survived', 'Sex']].isna().any(axis=1).sum())
"""),
        md(r"""
### Boolean masks — the building blocks

A mask is a `True`/`False` column. Inside the brackets it *filters* rows; on its own its `.mean()` is a
probability. We'll lean on three masks (and the `Survived` column, which is already 0/1).
"""),
        code(r"""
female   = titanic['Sex'] == 'female'      # True/False, one per passenger
male     = titanic['Sex'] == 'male'
survived = titanic['Survived'] == 1        # same idea, as an explicit mask

print('type of `female` mask :', type(female).__name__)
print('first 5 values        :', female.head().to_list())
print(f'\nfemale passengers : {female.sum()} of {len(titanic)}')
print(f'male passengers   : {male.sum()} of {len(titanic)}')
print(f'survivors         : {survived.sum()} of {len(titanic)}')
"""),
        md(r"""
## 3) Build it step by step

Now the heart of it. We'll lay down **one grid of counts** with `pd.crosstab`, then pull each kind of
probability out of it by hand — and check every single one against a direct boolean-mask calculation with
`np.isclose`, printing `match? True`.

### The count table

`pd.crosstab(rows, cols)` counts how many passengers fall in each combination. We put `Survived` down the
rows and `Sex` across the columns. Each inner cell is a count of people; the margins are the totals.
"""),
        code(r"""
counts = pd.crosstab(titanic['Survived'], titanic['Sex'])
print('COUNTS — how many passengers fall in each box:')
print(counts)

# Pull out the four cells + the totals we'll divide with, all from this grid.
grand_total  = counts.to_numpy().sum()           # everyone = 887
survived_row = counts.loc[1].sum()               # row total: all survivors
female_col   = counts['female'].sum()            # column total: all women
male_col     = counts['male'].sum()              # column total: all men
surv_female  = counts.loc[1, 'female']           # inner cell: survived AND female
surv_male    = counts.loc[1, 'male']             # inner cell: survived AND male

print(f'\ngrand total (everyone)   : {grand_total}')
print(f'survivors (a row total)  : {survived_row}')
print(f'women (a column total)   : {female_col}')
print(f'survived & female (cell) : {surv_female}')
"""),
        md(r"""
### Marginal probability — a total ÷ the grand total

A **marginal** probability ignores the other variable: it's an *edge* total over the grand total.
$P(\text{survived})$ is "all survivors ÷ everyone" — the survivor row total divided by 887. Check it against
the direct route, which is just the mean of the 0/1 `Survived` column.

$$P(\text{survived}) = \frac{\text{number who survived}}{\text{total number of passengers}}.$$
"""),
        code(r"""
p_survived_table  = survived_row / grand_total      # row total / grand total
p_survived_direct = titanic['Survived'].mean()      # mean of the 0/1 column

print(f'P(survived) from table  : {p_survived_table:.6f}')
print(f'P(survived) direct mask : {p_survived_direct:.6f}')
print('match?', np.isclose(p_survived_table, p_survived_direct))
print(f'\n-> about {p_survived_table*100:.0f}% of the 887 passengers survived.')
"""),
        md(r"""
### Joint probability — an inner cell ÷ the grand total

A **joint** probability is two things true at once: an *inner* cell over the grand total.
$P(\text{survived and female})$ is the "survived & female" cell divided by 887. The direct route is the mean
of the **combined** mask `survived & female`.

$$P(\text{survived and female}) = \frac{\text{number who survived and were female}}{\text{total}}.$$
"""),
        code(r"""
p_joint_table  = surv_female / grand_total           # inner cell / grand total
p_joint_direct = (survived & female).mean()          # mean of the combined mask

print(f'P(survived and female) from table  : {p_joint_table:.6f}')
print(f'P(survived and female) direct mask : {p_joint_direct:.6f}')
print('match?', np.isclose(p_joint_table, p_joint_direct))
"""),
        md(r"""
### Conditional probability — an inner cell ÷ a *column* total

A **conditional** probability shrinks the world to a subgroup. To get $P(\text{survived} \mid \text{female})$
we take the "survived & female" cell and divide by the **column** total for women — not the grand total. That
re-scales the cell to the world of women only. The direct route restricts to women first
(`titanic.loc[female, 'Survived']`), then takes the survival rate.

$$P(\text{survived} \mid \text{female}) = \frac{\text{survived and female}}{\text{all females}}
   = \frac{P(\text{survived and female})}{P(\text{female})}.$$
"""),
        code(r"""
p_cond_female_table  = surv_female / female_col                      # cell / column total
p_cond_female_direct = titanic.loc[female, 'Survived'].mean()        # restrict, then mean

p_cond_male_table  = surv_male / male_col
p_cond_male_direct = titanic.loc[male, 'Survived'].mean()

print(f'P(survived | female) from table  : {p_cond_female_table:.6f}')
print(f'P(survived | female) direct mask : {p_cond_female_direct:.6f}')
print('match?', np.isclose(p_cond_female_table, p_cond_female_direct))
print()
print(f'P(survived | male) from table    : {p_cond_male_table:.6f}')
print(f'P(survived | male) direct mask   : {p_cond_male_direct:.6f}')
print('match?', np.isclose(p_cond_male_table, p_cond_male_direct))
"""),
        md(r"""
**Read that out loud.** A woman's survival chance (~74%) is nearly **four times** a man's (~19%), and both are
miles from the overall ~39%. "Women and children first" isn't a legend here — it falls straight out of three
divisions on a grid of counts.
"""),
        md(r"""
### Independence check — does knowing sex change the bet?

Two events are **independent** if conditioning on one doesn't move the other:
$P(\text{survived} \mid \text{female}) = P(\text{survived})$. We can just *subtract* to see the gap. If it's
near zero they're independent; if it's big they're dependent. Sex and survival are about as dependent as it
gets.
"""),
        code(r"""
gap_female = p_cond_female_table - p_survived_table
gap_male   = p_cond_male_table   - p_survived_table

print(f'P(survived)          = {p_survived_table:.4f}')
print(f'P(survived | female) = {p_cond_female_table:.4f}   gap {gap_female:+.4f}')
print(f'P(survived | male)   = {p_cond_male_table:.4f}   gap {gap_male:+.4f}')
print()
print('Both gaps are huge, not ~0 -> sex and survival are strongly DEPENDENT.')
print(f'Knowing a passenger is female swings the survival bet up by {gap_female*100:+.0f} '
      f'percentage points; male, {gap_male*100:+.0f}.')
"""),
        md(r"""
### One step of Bayes — flip the conditioning

We have $P(\text{survived} \mid \text{female})$. The *backwards* question — among **survivors**, what fraction
were female, $P(\text{female} \mid \text{survived})$ — is a genuinely different number. On our grid the flip
is easy: divide the same "survived & female" cell by the **row** total (all survivors) instead of the column
total. Verify against the direct route: restrict to survivors, then take the female rate.
"""),
        code(r"""
p_female_given_surv_table  = surv_female / survived_row                  # cell / ROW total (flip!)
p_female_given_surv_direct = (titanic.loc[survived, 'Sex'] == 'female').mean()

print(f'P(survived | female)  = {p_cond_female_table:.4f}   (cell / column total)')
print(f'P(female | survived)  = {p_female_given_surv_table:.4f}   (cell / row total)')
print(f'direct check          = {p_female_given_surv_direct:.4f}')
print('match?', np.isclose(p_female_given_surv_table, p_female_given_surv_direct))
print()
print('Same cell, divided by a different total -> a different probability.')
print('P(survived | female) != P(female | survived): order matters. That is Bayes in one line.')
"""),
        md(r"""
## 4) The idiom

### A 0/1 column's `groupby(...).mean()` *is* the conditional probability

Everything above was hand division. The pandas idiom packs it into one expression: group the 0/1 `Survived`
column by `Sex` and take the mean of each group. Because the mean of a 0/1 column is a survival rate,
`groupby('Sex')['Survived'].mean()` gives you $P(\text{survived} \mid \text{sex})$ for every sex at once.
"""),
        code(r"""
by_sex = titanic.groupby('Sex')['Survived'].mean()
print('groupby(Sex)[Survived].mean():')
print(by_sex.round(6))
print()
print(f"P(survived | female): groupby = {by_sex['female']:.6f}, our by-hand = {p_cond_female_table:.6f}")
print(f"P(survived | male)  : groupby = {by_sex['male']:.6f}, our by-hand = {p_cond_male_table:.6f}")
print('match?', np.isclose(by_sex['female'], p_cond_female_table),
              np.isclose(by_sex['male'],   p_cond_male_table))
"""),
        md(r"""
### `crosstab(normalize=...)` reproduces the tables we built by hand

`pd.crosstab` will do the dividing for us. The `normalize` argument picks which total to divide by:

- `normalize='all'` → every cell ÷ the **grand total** = the **joint** probabilities.
- `normalize='columns'` → every cell ÷ its **column** total = $P(\text{survived} \mid \text{sex})$ (the
  column-conditional table — each column sums to 1).
- `normalize='index'` → every cell ÷ its **row** total = $P(\text{sex} \mid \text{survived})$ (the
  row-conditional table — each row sums to 1, this is the Bayes-flip table).
"""),
        code(r"""
joint_tbl = pd.crosstab(titanic['Survived'], titanic['Sex'], normalize='all')
print("normalize='all'  -> JOINT (each cell / grand total):")
print(joint_tbl.round(4))
print(f"  check: cell [survived=1, female] = {joint_tbl.loc[1, 'female']:.4f} "
      f"vs our joint {p_joint_table:.4f}  -> match? {np.isclose(joint_tbl.loc[1, 'female'], p_joint_table)}")

col_tbl = pd.crosstab(titanic['Survived'], titanic['Sex'], normalize='columns')
print("\nnormalize='columns'  -> P(survived | sex) (each column sums to 1):")
print(col_tbl.round(4))
print(f"  check: [survived=1, female] = {col_tbl.loc[1, 'female']:.4f} "
      f"vs P(survived|female) {p_cond_female_table:.4f}  -> match? "
      f"{np.isclose(col_tbl.loc[1, 'female'], p_cond_female_table)}")

row_tbl = pd.crosstab(titanic['Survived'], titanic['Sex'], normalize='index')
print("\nnormalize='index'  -> P(sex | survived) (each row sums to 1, the Bayes flip):")
print(row_tbl.round(4))
print(f"  check: [survived=1, female] = {row_tbl.loc[1, 'female']:.4f} "
      f"vs P(female|survived) {p_female_given_surv_table:.4f}  -> match? "
      f"{np.isclose(row_tbl.loc[1, 'female'], p_female_given_surv_table)}")
"""),
        md(r"""
That's the payoff: the three `normalize` modes are exactly the three divisions we did by hand (÷ grand total,
÷ column total, ÷ row total). Once you see *which total* each one divides by, `crosstab` stops being magic.
"""),
        md(r"""
## 5) Refactor into a reusable function

We computed conditional probabilities the same way three times: restrict to the rows where a "given" column
equals some value, then take the rate of the event. The programmer's reflex is to **wrap repeated work in a
function** — name it, give it inputs, `return` the result. Then *any* conditional probability is one call,
on any two columns.
"""),
        code(r"""
def cond_prob(df, event_col, event_val, given_col, given_val):
    '''P(event_col == event_val | given_col == given_val) by restricting then averaging.'''
    given = df[given_col] == given_val          # mask: the subgroup we condition on
    sub = df.loc[given]                         # shrink the world to that subgroup
    return (sub[event_col] == event_val).mean() # fraction of the subgroup with the event

# Reuse it to reproduce conditionals from Section 3 — no copy-pasted division:
p1 = cond_prob(titanic, 'Survived', 1, 'Sex', 'female')
p2 = cond_prob(titanic, 'Survived', 1, 'Sex', 'male')
p3 = cond_prob(titanic, 'Sex', 'female', 'Survived', 1)   # the Bayes flip, same function

print(f"cond_prob(survived=1 | female)   = {p1:.6f}   (matches §3? {np.isclose(p1, p_cond_female_table)})")
print(f"cond_prob(survived=1 | male)     = {p2:.6f}   (matches §3? {np.isclose(p2, p_cond_male_table)})")
print(f"cond_prob(female   | survived=1) = {p3:.6f}   (matches §3? {np.isclose(p3, p_female_given_surv_table)})")
"""),
        md(r"""
A function is a contract: *give me a table, an event, and a condition — I'll give you the probability.* Once
it's correct you stop thinking about which total to divide by and start thinking in **events and conditions**
— exactly the leap from "writing probabilities" to "doing probability in code."
"""),
        md(r"""
## 6) Now you code

Write the code yourself. Add a new cell (the **+** button in the toolbar) and try each — predict the answer
before you run it.

1. **A conditional on a third variable.** The `Pclass` column is the ticket class (1, 2, 3). Use your
   `cond_prob` to compute `cond_prob(titanic, 'Survived', 1, 'Pclass', 1)` — the survival rate in first class.
   Will it be above or below the overall ~0.39? Then try `Pclass` 3 and compare.

2. **Verify the multiplication rule.** $P(\text{survived and female})$ should equal
   $P(\text{survived} \mid \text{female}) \times P(\text{female})$. Compute the right-hand side
   (`cond_prob(...) * (titanic['Sex'] == 'female').mean()`) and check it against `(survived & female).mean()`
   with `np.isclose`. It should print `True`.

3. **Build the whole P(survived | sex) table two ways.** Compute `titanic.groupby('Sex')['Survived'].mean()`
   and `pd.crosstab(titanic['Survived'], titanic['Sex'], normalize='columns')`. Find the survived-row of the
   crosstab and confirm the two agree number-for-number.

4. **A second Bayes flip.** You have `P(female | survived)`. Now compute `P(survived | female)` *from it* with
   Bayes by hand: `(cond_prob(titanic,'Sex','female','Survived',1) * titanic['Survived'].mean()) /
   (titanic['Sex']=='female').mean()`. Check it lands back on the ~0.742 from Section 3.

## What you learned

**Coding**
- A **boolean mask** (`df['Sex'] == 'female'`) is a `True`/`False` column; inside brackets it filters rows, and
  the **mean of a 0/1 (or True/False) column is a probability** — the single most useful pattern here.
- A **contingency table** from `pd.crosstab` is a grid of counts; every probability is one cell or total
  divided by another (÷ grand total, ÷ column total, ÷ row total).
- `pd.crosstab(..., normalize='all' / 'columns' / 'index')` does that dividing for you — joint,
  column-conditional, and row-conditional tables respectively.
- A 0/1 column's `groupby(key).mean()` is $P(\text{event} \mid \text{key})$ for every group in one line.
- **Refactor repeated work into a function** with clear inputs and a `return`, then reuse it everywhere.

**Statistics (now demystified)**
- **Marginal** = edge total ÷ grand total; **joint** = inner cell ÷ grand total; **conditional** = inner cell
  ÷ a row/column total. You reproduced every Lesson 5 number to the decimal, each checked with `np.isclose`.
- **Independence** is just $P(A \mid B) = P(A)$ — compare the conditional to the marginal; survival vs. sex
  fails it enormously.
- **Bayes flips the conditioning** by dividing the same cell by a *different* total:
  $P(A \mid B) \ne P(B \mid A)$ in general.

**Back to the core lesson** ([Lesson 5: Probability & conditional probability](05-probability.ipynb)) for the
*meaning* of these numbers, or read on through the course.
"""),
        md(nav_md(ID)),
    ]


if __name__ == "__main__":
    save_lesson(ID, cells())
