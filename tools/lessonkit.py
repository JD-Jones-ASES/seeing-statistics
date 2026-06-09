r"""
lessonkit.py — shared machinery for every lesson in the course.

ONE source of truth (the MANIFEST below) drives three things so they can never
drift apart:
  * the notebook build (tools/build_notebooks.py),
  * the consistent header + previous/next navigation inside each lesson,
  * the course landing page (tools/make_index.py).

Each lesson exists in up to three VARIANTS, all sharing one `num`:
  * "core"  — the main lesson (math -> meaning -> interpretation, on real data),
  * "data"  — the same ideas on a *different* dataset (see what changes),
  * "code"  — the same ideas with the focus on the Python/pandas/plotting craft.
The "core" variants, in list order, are the reading spine (prev/next flow through
them); the "data"/"code" variants hang off their core lesson and link back to it.

Authoring conventions:
  * The MANIFEST `module` and `file` fields are AUTHORITATIVE. A module's filename
    need not match its display number. Legacy lessons keep numeric module names
    (lesson_00..07, lesson_flagship) and integer `num` keys; new lessons use
    slug module names (e.g. lesson_relationships) and string `id`s.
  * A lesson module exposes cells() and addresses itself by its key — an int
    `num` (legacy) or a string `id` (new):
        from lessonkit import md, code, save_lesson, header_md, nav_md
        ID = "relationships"
        def cells(): return [md(header_md(ID, intro="...")), ...., md(nav_md(ID))]
        if __name__ == "__main__": save_lesson(ID, cells())

Design rules honored: datasets in raw/ stay read-only (we only ever read them);
nothing here deletes or rewrites anything.
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent       # statistics-course/
LESSONS = ROOT / "lessons"
LESSONS.mkdir(exist_ok=True)

KERNEL = {"name": "stats-course", "display_name": "Python (stats course)", "language": "python"}

# Section names (NO "semester" language — the project uses "Part").
PART_NAMES = {
    1: "Seeing data & the logic of chance",
    2: "Drawing conclusions from data",
}

# ---------------------------------------------------------------------------
# THE MANIFEST — the spine of the whole course. List order is reading order.
# Fields: id (unique str), num (unique-among-core int key), variant
# (core|data|code), part (1|2), module (source module under lessons_src/),
# file (.ipynb name), title, data (human blurb of dataset), blurb (one line).
# Optional flags: orientation=True (Lesson 0), flagship=True.
# A "data"/"code" variant shares its core's `num`; add `parent` = core id.
#
# Variant FILENAME convention: a variant's file is its core's stem + a "~"
# separator + the variant tag, e.g. core "01-describing-data.ipynb" ->
# "01-describing-data~another-dataset.ipynb" and "...~the-code.ipynb". The "~" is
# deliberate and was chosen empirically. JupyterLab's file browser sorts with
# locale collation (Intl.Collator), and under it the FIRST char after the shared
# stem decides order: the core has "." (from ".ipynb") and the variant has the
# separator. "-", "_", ".", and space ALL collate before "." there, which floats
# the companion ABOVE its core (and the descriptor letters are never reached). "~"
# is one of the few readable, filename/URL-safe characters that collates AFTER "."
# in BOTH locale collation and plain byte order, so the core lists first with its
# companions right beneath it (and "another-dataset" < "the-code" keeps data
# before code). Verified with Intl.Collator. Keep "~" (never "-"/"_") for variants.
# ---------------------------------------------------------------------------
MANIFEST = [
    # ---- Part 1 — Seeing data & the logic of chance ----
    dict(id="00", num=0, variant="core", part=1, module="lesson_00",
         file="00-how-these-notebooks-work.ipynb",
         title="How these notebooks work (start here)",
         data="no data yet — just orientation", orientation=True,
         blurb="Never used a notebook? Five minutes here and you'll know how to run a lesson, "
               "re-run safely, and read the output. No statistics yet."),
    dict(id="01", num=1, variant="core", part=1, module="lesson_01",
         file="01-describing-data.ipynb",
         title="Describing a real dataset",
         data="Palmer Penguins",
         blurb="Center, spread, shape, correlation and z-scores — the first questions to ask of any data."),
    dict(id="01-data", num=1, variant="data", part=1, parent="01", module="lesson_01_data",
         file="01-describing-data~another-dataset.ipynb",
         title="Describing a different dataset: Fisher's irises",
         data="Iris flowers (150 blooms, 3 species)",
         blurb="The same first questions on Fisher's irises — where the groups are tight and clean, and one pair of measurements is almost a straight line."),
    dict(id="01-code", num=1, variant="code", part=1, parent="01", module="lesson_01_code",
         file="01-describing-data~the-code.ipynb",
         title="The code behind it: build describe() from scratch",
         data="Palmer Penguins (same as Lesson 1)",
         blurb="Open the hood on Lesson 1 — compute the mean, SD, z-scores and correlation yourself in NumPy, meet split-apply-combine with groupby, then wrap it all in your own describe() function."),
    dict(id="02", num=2, variant="core", part=1, module="lesson_02",
         file="02-distributions-shape-outliers.ipynb",
         title="Distributions: shape, skew & outliers",
         data="Old Faithful · Lending Club",
         blurb="Frequency tables, histograms, skew, the IQR outlier rule, and z-scores — when a single number lies."),
    dict(id="02-data", num=2, variant="data", part=1, parent="02", module="lesson_02_data",
         file="02-distributions-shape-outliers~another-dataset.ipynb",
         title="Shape & skew on a different dataset: world life expectancy",
         data="Gapminder life expectancy, 2007",
         blurb="The same tools on the world's life expectancies — a distribution that is *left*-skewed, so the long tail (and the skew sign) point the other way."),
    dict(id="02-code", num=2, variant="code", part=1, parent="02", module="lesson_02_code",
         file="02-distributions-shape-outliers~the-code.ipynb",
         title="The code behind it: histograms, skew & the IQR rule in code",
         data="Old Faithful · Lending Club (same as Lesson 2)",
         blurb="Lesson 2's shape tools in code — build a frequency table and a histogram from scratch, compute skew and the 1.5×IQR outlier fence yourself, then wrap the fence as a reusable outlier-flagging function."),
    dict(id="relationships", num=11, variant="core", part=1, module="lesson_relationships",
         file="03-relationships-regression.ipynb",
         title="Relationships: correlation & the regression line",
         data="Ames housing",
         blurb="Scatterplots, the correlation r, the least-squares line, residuals & R² — and why "
               "correlation isn't causation."),
    dict(id="relationships-data", num=11, variant="data", part=1, parent="relationships",
         module="lesson_relationships_data",
         file="03-relationships-regression~another-dataset.ipynb",
         title="Relationships on a different dataset: wealth & life expectancy",
         data="Gapminder GDP vs life expectancy, 2007",
         blurb="Wealth vs. life expectancy across 142 countries — a clearly *curved* relationship a straight "
               "line gets wrong, until a log transform straightens it out."),
    dict(id="relationships-code", num=11, variant="code", part=1, parent="relationships",
         module="lesson_relationships_code",
         file="03-relationships-regression~the-code.ipynb",
         title="The code behind it: fit a regression line by hand",
         data="Ames housing (same as Lesson 3)",
         blurb="Build Lesson 3's regression line in code — compute Pearson's r and the least-squares slope & "
               "intercept straight from their formulas with NumPy, draw the fit and its residuals, then wrap it "
               "as your own fit_line() and check it against np.polyfit."),
    dict(id="study-design", num=12, variant="core", part=1, module="lesson_study_design",
         file="04-where-data-comes-from.ipynb",
         title="Where data comes from: sampling & study design",
         data="simulation + a known population",
         blurb="Sampling methods, bias, experiments vs. observation, random assignment & confounding — "
               "how trustworthy data is made."),
    dict(id="03", num=3, variant="core", part=1, module="lesson_03",
         file="05-probability.ipynb",
         title="Probability & conditional probability",
         data="Titanic",
         blurb="Marginal, joint and conditional probability, independence, and a first taste of Bayes — "
               "counted from real survivors."),
    dict(id="03-data", num=3, variant="data", part=1, parent="03", module="lesson_03_data",
         file="05-probability~another-dataset.ipynb",
         title="Conditional probability on a different dataset: Palmer penguins",
         data="Palmer Penguins",
         blurb="The same probability rules counted from penguins — including a pair of traits that turn out "
               "to be nearly *independent*, the opposite of Titanic's strong dependence."),
    dict(id="03-code", num=3, variant="code", part=1, parent="03", module="lesson_03_code",
         file="05-probability~the-code.ipynb",
         title="The code behind it: probabilities from a crosstab",
         data="Titanic (same as Lesson 5)",
         blurb="Lesson 5's probabilities straight from the data — build a two-way table with pd.crosstab, read "
               "marginal, joint and conditional probabilities off it with boolean masks, and write your own "
               "conditional-probability function."),
    dict(id="counting", num=13, variant="core", part=1, module="lesson_counting",
         file="06-counting.ipynb",
         title="Counting: permutations & combinations",
         data="dice & cards (simulation)",
         blurb="Permutations, combinations and the choose function — the counting that powers the binomial."),
    dict(id="04", num=4, variant="core", part=1, module="lesson_04",
         file="07-random-variables.ipynb",
         title="Random variables, expectation & variance",
         data="dice simulation · US Births 2014",
         blurb="Turn outcomes into numbers: probability distributions, expected value, variance, and the "
               "law of large numbers."),
    dict(id="04-data", num=4, variant="data", part=1, parent="04", module="lesson_04_data",
         file="07-random-variables~another-dataset.ipynb",
         title="A random variable from real data: how many relatives aboard?",
         data="Titanic",
         blurb="A genuinely lopsided real discrete random variable — the number of family members each Titanic "
               "passenger travelled with — with its expectation, variance, and the law of large numbers."),
    dict(id="04-code", num=4, variant="code", part=1, parent="04", module="lesson_04_code",
         file="07-random-variables~the-code.ipynb",
         title="The code behind it: simulate a random variable",
         data="dice simulation · US Births 2014 (same as Lesson 7)",
         blurb="Lesson 7 as a simulation — use a NumPy random generator to roll dice, build a probability "
               "distribution and compute its expectation & variance from arrays, and watch the law of large "
               "numbers converge as a running average settles down."),
    dict(id="05", num=5, variant="core", part=1, module="lesson_05",
         file="08-binomial-poisson.ipynb",
         title="Binomial & Poisson — counting events",
         data="US Births 2014 · USGS earthquakes",
         blurb="Two workhorse models for counts: successes out of n (binomial) and rare events per interval (Poisson)."),
    dict(id="05-data", num=5, variant="data", part=1, parent="05", module="lesson_05_data",
         file="08-binomial-poisson~another-dataset.ipynb",
         title="Binomial & Poisson on different data: survival & quakes",
         data="Titanic · USGS earthquakes",
         blurb="The binomial as survivors out of n aboard the Titanic, and the Poisson on rarer (M5+) earthquakes "
               "per day — the same two count models at different rates."),
    dict(id="05-code", num=5, variant="code", part=1, parent="05", module="lesson_05_code",
         file="08-binomial-poisson~the-code.ipynb",
         title="The code behind it: build the binomial & Poisson in NumPy",
         data="US Births 2014 · USGS earthquakes (same as Lesson 8)",
         blurb="Build Lesson 8's two count models in code — simulate successes-out-of-n and rare events with a "
               "random generator, compare your simulated histogram against the formula, and wrap each "
               "probability-mass function as a reusable function checked against SciPy."),
    dict(id="06", num=6, variant="core", part=1, module="lesson_06",
         file="09-normal-distribution.ipynb",
         title="The normal distribution & the empirical rule",
         data="Galton family heights",
         blurb="The bell curve, the 68–95–99.7 rule, and turning z-scores into probabilities — checked on real heights."),
    dict(id="06-data", num=6, variant="data", part=1, parent="06", module="lesson_06_data",
         file="09-normal-distribution~another-dataset.ipynb",
         title="The bell curve on a different dataset: 5,000 adult heights",
         data="NHANES adult-male heights",
         blurb="The empirical rule on a much larger modern sample (~5,000 men) — where the 68–95–99.7 bands "
               "land almost exactly on the textbook numbers."),
    dict(id="06-code", num=6, variant="code", part=1, parent="06", module="lesson_06_code",
         file="09-normal-distribution~the-code.ipynb",
         title="The code behind it: the empirical rule in code",
         data="Galton family heights (same as Lesson 9)",
         blurb="Lesson 9's bell curve in code — standardize heights to z-scores yourself, count the share inside "
               "±1/±2/±3 SDs with boolean masks, and overlay the normal curve on a histogram you scale to match "
               "by hand, then wrap it as an empirical_rule() check."),
    dict(id="07", num=7, variant="core", part=1, module="lesson_07",
         file="10-sampling-distributions-clt.ipynb",
         title="Sampling distributions & the Central Limit Theorem",
         data="Lending Club · simulation",
         blurb="The engine of the whole course: averages of samples become normal — even when the data is wildly skewed."),
    dict(id="07-data", num=7, variant="data", part=1, parent="07", module="lesson_07_data",
         file="10-sampling-distributions-clt~another-dataset.ipynb",
         title="The CLT on a different dataset: Ames sale prices",
         data="Ames housing sale price",
         blurb="The same engine on house *prices* — a differently-skewed population, showing how the shape of "
               "the data sets how big a sample the CLT needs."),
    dict(id="07-code", num=7, variant="code", part=1, parent="07", module="lesson_07_code",
         file="10-sampling-distributions-clt~the-code.ipynb",
         title="The code behind it: simulate the CLT yourself",
         data="Lending Club · simulation (same as Lesson 10)",
         blurb="Simulate Lesson 10's Central Limit Theorem yourself — draw thousands of samples (a loop first, "
               "then the vectorized idiom), build the sampling distribution of the mean, and watch it turn "
               "normal as n grows while its standard error shrinks by the √n law."),

    # ---- Part 2 — Drawing conclusions from data ----
    dict(id="estimation", num=14, variant="core", part=2, module="lesson_estimation",
         file="11-estimation.ipynb",
         title="Estimation & standard error",
         data="Ames housing",
         blurb="Point estimates, bias vs. precision, and the standard error — how far a sample statistic typically "
               "lands from the truth, and the √n law that shrinks it."),
    dict(id="estimation-data", num=14, variant="data", part=2, parent="estimation", module="lesson_estimation_data",
         file="11-estimation~another-dataset.ipynb",
         title="Estimation on a different population: adult heights",
         data="NHANES adult-male heights",
         blurb="The same standard-error story on a large, near-normal population (~5,000 heights) instead of skewed "
               "house sizes — unbiasedness, the standard error, and the √n law on tidy data."),
    dict(id="estimation-code", num=14, variant="code", part=2, parent="estimation", module="lesson_estimation_code",
         file="11-estimation~the-code.ipynb",
         title="The code behind it: standard error by simulation",
         data="Ames housing (same as Lesson 11)",
         blurb="Build Lesson 11's standard-error story in code — treat Ames as the population, simulate the sampling "
               "distribution to show SE = σ/√n, watch the √n law and unbiasedness, then estimate the SE from a single "
               "sample as s/√n."),
    dict(id="flagship", num=9, variant="core", part=2, module="lesson_confidence_intervals",
         file="12-confidence-intervals.ipynb",
         title="What a 95% confidence interval really means",
         data="Ames, Iowa housing",
         blurb="The flagship simulation: build 100 confidence intervals from real data and watch ~95 capture the true mean.",
         flagship=True),
    dict(id="flagship-data", num=9, variant="data", part=2, parent="flagship", module="lesson_confidence_intervals_data",
         file="12-confidence-intervals~another-dataset.ipynb",
         title="Confidence intervals on a more skewed variable: sale price",
         data="Ames housing sale price",
         blurb="The 100-interval coverage demo again, now on the more strongly skewed sale price — does 95% "
               "coverage still hold when the population is even less bell-shaped?"),
    dict(id="flagship-code", num=9, variant="code", part=2, parent="flagship", module="lesson_confidence_intervals_code",
         file="12-confidence-intervals~the-code.ipynb",
         title="The code behind it: build the CI & its coverage demo",
         data="Ames, Iowa housing (same as Lesson 12)",
         blurb="The flagship coverage demo, coded from scratch — build a 95% CI by the t-formula (checked against "
               "SciPy), then simulate hundreds of intervals and count how many capture the true mean (~95), with a "
               "confidence-level sweep and z-vs-t."),
    dict(id="hypothesis-testing", num=15, variant="core", part=2, module="lesson_hypothesis",
         file="13-hypothesis-testing.ipynb",
         title="Hypothesis testing: p-values, errors & power",
         data="simulation on a known population",
         blurb="Null vs. alternative, the p-value, significance, Type I/II errors, and the power of a test — built "
               "by simulating the null world."),
    dict(id="hypothesis-testing-data", num=15, variant="data", part=2, parent="hypothesis-testing",
         module="lesson_hypothesis_data",
         file="13-hypothesis-testing~another-dataset.ipynb",
         title="Hypothesis testing on real data: do penguin species differ?",
         data="Palmer Penguins",
         blurb="The null world built from real penguin weights — a true near-null (two species that barely differ) "
               "and an obvious effect (a third that clearly does), with Type I error and power made visible."),
    dict(id="hypothesis-testing-code", num=15, variant="code", part=2, parent="hypothesis-testing",
         module="lesson_hypothesis_code",
         file="13-hypothesis-testing~the-code.ipynb",
         title="The code behind it: build a test by simulating the null",
         data="simulation on a known population (same as Lesson 13)",
         blurb="Build a hypothesis test by simulating the null world — get a p-value as the fraction of chance "
               "outcomes at least as extreme (checked against SciPy), then watch the Type I rate sit near 5% and "
               "power rise with effect size and n."),
    dict(id="proportions", num=16, variant="core", part=2, module="lesson_proportions",
         file="14-inference-proportions.ipynb",
         title="Inference for proportions (one & two)",
         data="US Births 2014",
         blurb="Confidence intervals and tests for one proportion and for the difference of two — counted from real births."),
    dict(id="proportions-data", num=16, variant="data", part=2, parent="proportions", module="lesson_proportions_data",
         file="14-inference-proportions~another-dataset.ipynb",
         title="Inference for proportions on different data: who survived?",
         data="Titanic",
         blurb="One- and two-proportion intervals and tests on Titanic survival — a difference so large (women vs. "
               "men) the conclusion is never in doubt, the mirror of a borderline case."),
    dict(id="proportions-code", num=16, variant="code", part=2, parent="proportions", module="lesson_proportions_code",
         file="14-inference-proportions~the-code.ipynb",
         title="The code behind it: proportion inference from counts",
         data="US Births 2014 (same as Lesson 14)",
         blurb="Lesson 14's proportion inference from counts — build p̂, its standard error and one- and two-proportion "
               "z-tests & CIs from the formulas, checked against statsmodels, with the mean-of-a-0/1-column = a "
               "proportion idiom."),
    dict(id="t-tests", num=17, variant="core", part=2, module="lesson_ttests",
         file="15-t-tests.ipynb",
         title="t-tests: one-sample, two-sample & paired",
         data="ToothGrowth · Swim (paired)",
         blurb="Comparing means with the t-distribution: against a target, between two groups, and within matched pairs."),
    dict(id="t-tests-data", num=17, variant="data", part=2, parent="t-tests", module="lesson_ttests_data",
         file="15-t-tests~another-dataset.ipynb",
         title="t-tests on different data: chick feeds & textbook prices",
         data="Chickwts · Textbooks (paired)",
         blurb="A clean two-sample test (two chicken feeds) and a different paired design (the same textbook priced "
               "at the campus store vs. online) — comparing means on fresh data."),
    dict(id="t-tests-code", num=17, variant="code", part=2, parent="t-tests", module="lesson_ttests_code",
         file="15-t-tests~the-code.ipynb",
         title="The code behind it: build the t-statistic from scratch",
         data="ToothGrowth · Swim (paired) (same as Lesson 15)",
         blurb="Build the t-statistic from scratch — one-sample, Welch two-sample, and the paired trick (reduce pairs "
               "to differences), each checked against scipy.stats.ttest_*, on ToothGrowth and the paired swim data."),
    dict(id="anova", num=18, variant="core", part=2, module="lesson_anova",
         file="16-anova.ipynb",
         title="Comparing many groups: ANOVA",
         data="Palmer Penguins",
         blurb="One-way ANOVA and the F-test — is at least one group mean different, without inflating error by "
               "testing every pair?"),
    dict(id="anova-data", num=18, variant="data", part=2, parent="anova", module="lesson_anova_data",
         file="16-anova~another-dataset.ipynb",
         title="ANOVA on a different dataset: six chicken feeds",
         data="Chickwts",
         blurb="One-way ANOVA across six feed supplements — more groups than the penguins, a classic designed "
               "experiment where the F-test shows at least one feed really differs."),
    dict(id="anova-code", num=18, variant="code", part=2, parent="anova", module="lesson_anova_code",
         file="16-anova~the-code.ipynb",
         title="The code behind it: ANOVA from the variance decomposition",
         data="Palmer Penguins (same as Lesson 16)",
         blurb="Build one-way ANOVA from the variance decomposition — split total into between- and within-group "
               "sums of squares (they add up), form the mean squares and the F-ratio, get the p-value, all checked "
               "against scipy.stats.f_oneway."),
    dict(id="chi-square", num=19, variant="core", part=2, module="lesson_chisquare",
         file="17-chi-square.ipynb",
         title="Chi-square: goodness-of-fit & independence",
         data="Titanic",
         blurb="Testing counts: does a distribution match what we expected, and are two categorical variables associated?"),
    dict(id="chi-square-data", num=19, variant="data", part=2, parent="chi-square", module="lesson_chisquare_data",
         file="17-chi-square~another-dataset.ipynb",
         title="Chi-square on different data: penguin species & island",
         data="Palmer Penguins",
         blurb="A chi-square test of independence where the association is near-total — some penguin species live "
               "on only one island — the opposite extreme from a borderline table."),
    dict(id="chi-square-code", num=19, variant="code", part=2, parent="chi-square", module="lesson_chisquare_code",
         file="17-chi-square~the-code.ipynb",
         title="The code behind it: chi-square from a crosstab",
         data="Titanic (same as Lesson 17)",
         blurb="Build the chi-square test from a crosstab — expected counts under independence by an outer product, "
               "χ² = Σ(O−E)²/E vectorized over the table, the df and p-value, checked against "
               "scipy.stats.chi2_contingency."),
    dict(id="regression-inference", num=20, variant="core", part=2, module="lesson_regression_inference",
         file="18-regression-inference.ipynb",
         title="Correlation & regression: inference",
         data="Ames housing",
         blurb="Is the slope real, or just luck? A confidence interval and test for a regression slope, with "
               "residual diagnostics."),
    dict(id="regression-inference-data", num=20, variant="data", part=2, parent="regression-inference",
         module="lesson_regression_inference_data",
         file="18-regression-inference~another-dataset.ipynb",
         title="Regression inference on different data: wealth & longevity",
         data="Gapminder (log GDP vs life expectancy)",
         blurb="Is the slope real? A confidence interval and t-test for the regression slope on the log-wealth vs. "
               "life-expectancy fit, with residual diagnostics on a curved-then-straightened relationship."),
    dict(id="regression-inference-code", num=20, variant="code", part=2, parent="regression-inference",
         module="lesson_regression_inference_code",
         file="18-regression-inference~the-code.ipynb",
         title="The code behind it: inference for a regression slope",
         data="Ames housing (same as Lesson 18)",
         blurb="Build inference for a regression slope — the residual standard error, the slope's standard error, its "
               "t-statistic, p-value and confidence interval from the formulas, checked against "
               "scipy.stats.linregress, with residual diagnostics."),
    dict(id="multiple-regression", num=21, variant="core", part=2, module="lesson_multiple_regression",
         file="19-multiple-regression.ipynb",
         title="Multiple regression (intro)",
         data="Ames housing",
         blurb="Several predictors at once: adjusted ('holding others constant') effects, and how to read a regression table."),
    dict(id="multiple-regression-data", num=21, variant="data", part=2, parent="multiple-regression",
         module="lesson_multiple_regression_data",
         file="19-multiple-regression~another-dataset.ipynb",
         title="Multiple regression on different data: penguin body mass",
         data="Palmer Penguins",
         blurb="Predicting penguin mass from flipper length — and watching the flipper slope change once species is "
               "added, the textbook picture of an adjusted ('holding species constant') effect."),
    dict(id="multiple-regression-code", num=21, variant="code", part=2, parent="multiple-regression",
         module="lesson_multiple_regression_code",
         file="19-multiple-regression~the-code.ipynb",
         title="The code behind it: multiple regression with linear algebra",
         data="Ames housing (same as Lesson 19)",
         blurb="Build multiple regression with linear algebra — assemble the design matrix and solve for the "
               "coefficients (β via np.linalg.lstsq), compute R² and adjusted R², check against statsmodels, and "
               "watch a slope change as predictors are added."),
    dict(id="bootstrap", num=22, variant="core", part=2, module="lesson_bootstrap",
         file="20-bootstrap-permutation.ipynb",
         title="The bootstrap & permutation tests",
         data="Old Faithful · ToothGrowth",
         blurb="Inference by resampling: a bootstrap confidence interval and a permutation test — honest answers "
               "when the textbook formula runs out."),
    dict(id="bootstrap-data", num=22, variant="data", part=2, parent="bootstrap", module="lesson_bootstrap_data",
         file="20-bootstrap-permutation~another-dataset.ipynb",
         title="Bootstrap & permutation on different data: wetsuits",
         data="Swim velocities (paired, n=12)",
         blurb="Resampling inference on a tiny paired sample — a bootstrap interval for the wetsuit speed boost and "
               "a permutation test — exactly where the textbook formula is shakiest."),
    dict(id="bootstrap-code", num=22, variant="code", part=2, parent="bootstrap", module="lesson_bootstrap_code",
         file="20-bootstrap-permutation~the-code.ipynb",
         title="The code behind it: build the bootstrap & permutation test",
         data="Old Faithful · ToothGrowth (same as Lesson 20)",
         blurb="Build resampling inference yourself — a bootstrap percentile CI by resampling with replacement, and "
               "a permutation test by shuffling group labels, each checked against the textbook formula, exactly "
               "where formulas run out."),
]

# ---------------------------------------------------------------------------
# Lookups & reading-order helpers
# ---------------------------------------------------------------------------
_BY_ID = {m["id"]: m for m in MANIFEST}
_BY_NUM = {m["num"]: m for m in MANIFEST if m["variant"] == "core"}   # core only (variants share num)
_SPINE = [m for m in MANIFEST if m["variant"] == "core"]             # reading order incl. orientation
_NUMBERED = [m for m in _SPINE if not m.get("orientation")]          # numbered "Lesson k of N"
COURSE_TOTAL = len(_NUMBERED)


def meta(key):
    """Resolve a manifest entry by string id (any variant) or int num (core)."""
    return _BY_ID[key] if isinstance(key, str) else _BY_NUM[key]


def _core_of(m):
    """The core entry sharing this entry's num (for a data/code variant)."""
    if m["variant"] == "core":
        return m
    return next(c for c in _SPINE if c["num"] == m["num"])


def _lesson_pos(m):
    """1-based display number of a lesson within the whole numbered course."""
    return _NUMBERED.index(_core_of(m)) + 1


def _variants_of(m):
    """The data/code variant entries attached to a core entry, in a stable order."""
    order = {"data": 0, "code": 1}
    sibs = [v for v in MANIFEST if v["variant"] != "core" and v["num"] == m["num"]]
    return sorted(sibs, key=lambda v: order.get(v["variant"], 9))


_VARIANT_TAG = {"data": "another dataset", "code": "the code behind it"}


def _position_label(m):
    """e.g. 'Part 1 · Lesson 3 of 20' or 'Part 1 · Lesson 1 · the code behind it'."""
    if m.get("orientation"):
        return "Start here · no statistics yet — just how the notebooks work"
    pos = _lesson_pos(m)
    if m["variant"] != "core":
        return f"Part {m['part']} · Lesson {pos} · {_VARIANT_TAG.get(m['variant'], m['variant'])}"
    tail = " · the flagship demo" if m.get("flagship") else ""
    return f"Part {m['part']} · Lesson {pos} of {COURSE_TOTAL}{tail}"


def _reading_neighbors(m):
    """(previous, next) core entries around core `m` in the reading spine (None at the ends)."""
    i = _SPINE.index(m)
    prev = _SPINE[i - 1] if i > 0 else None
    nxt = _SPINE[i + 1] if i < len(_SPINE) - 1 else None
    return prev, nxt


def _companion_links(core, prefix="**Same idea, another way:**"):
    """'<prefix> Another dataset → · The code behind it →' for a core's variants ('' if none)."""
    extras = _variants_of(core)
    if not extras:
        return ""
    links = "  ·  ".join(
        f"[{_VARIANT_TAG[v['variant']].capitalize()} →]({v['file']})" for v in extras)
    return f"{prefix}  {links}"


# ---------------------------------------------------------------------------
# Cell builders
# ---------------------------------------------------------------------------
def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def setup_note():
    """A standard, friendly 'how to run this' reassurance for newcomers.

    Auto-inserted right after each lesson's title (see save_lesson), so every
    lesson reassures a never-used-Jupyter reader in the same words.
    """
    return (
        "> **New to notebooks?** The grey boxes below are **code**. Click one and press "
        "**Shift + Enter** to run it — the result (a number, a table, or a chart) appears right "
        "underneath. Run them **top to bottom**. You do **not** need to understand the first "
        "*\"Setup\"* cell — it just loads the tools and the data, so run it and move on. And you "
        "**can't break anything**: re-running a cell is always safe. *(Brand new to all this? The "
        "five-minute [Lesson 0 — How these notebooks work](00-how-these-notebooks-work.ipynb) "
        "shows you the basics first.)*"
    )


def save_lesson(key, cells):
    """Write lessons/<file> for the entry `key` (no outputs; execution is separate).

    For every real lesson (not the orientation page) we slip the standard
    `setup_note()` in right after the title cell, so the 'how to run this'
    reassurance is identical and present everywhere without editing each lesson.
    """
    m = meta(key)
    if not m.get("orientation") and len(cells) > 1:
        cells = [cells[0], md(setup_note()), *cells[1:]]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = KERNEL
    nb.metadata["language_info"] = {"name": "python"}
    path = LESSONS / m["file"]
    nbf.write(nb, path)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


# ---------------------------------------------------------------------------
# Standard header + navigation (the UX glue)
# ---------------------------------------------------------------------------
def header_md(key, intro=""):
    """The opening markdown cell: title, a 'you are here' line, TOP navigation, and an intro.

    The top nav mirrors the footer so a reader can move on (or jump to a companion / the
    course map) without scrolling to the bottom: prev · map · next for a core lesson, and
    back-to-core · map · sibling for a data/code variant.
    """
    m = meta(key)
    if m.get("orientation"):
        heading = m["title"]
    elif m.get("flagship"):
        heading = f"Flagship demo — {m['title']}"
    elif m["variant"] != "core":
        heading = m["title"]
    else:
        heading = f"Lesson {_lesson_pos(m)} — {m['title']}"

    line = (f"# {heading}\n\n"
            f"*{_position_label(m)}  ·  data: {m['data']}*")

    if m["variant"] == "core":
        # Reading-spine navigation — now at the top as well as the bottom.
        prev, nxt = _reading_neighbors(m)
        nav = []
        if prev:
            nav.append(f"**←** [{_short(prev)}]({prev['file']})")
        nav.append("[↑ Course map](../rendered/index.html)")
        if nxt:
            nav.append(f"[{_short(nxt)}]({nxt['file']}) **→**")
        line += "\n\n" + "  ·  ".join(nav)
        comp = _companion_links(m)
        if comp:
            line += "\n\n" + comp
    else:
        # A variant: say what it companions, then link home / map / sibling.
        core = _core_of(m)
        line += (f"\n\n*↩ A companion to "
                 f"[Lesson {_lesson_pos(m)}: {core['title']}]({core['file']}).*")
        nav = [f"**↩ Back:** [{_short(core)}]({core['file']})",
               "[↑ Course map](../rendered/index.html)"]
        for sib in _variants_of(core):
            if sib["id"] != m["id"]:
                nav.append(f"**Also:** [{_VARIANT_TAG[sib['variant']].capitalize()} →]({sib['file']})")
        line += "\n\n" + "  ·  ".join(nav)

    if intro:
        line += "\n\n" + intro.strip("\n")
    return line


def nav_md(key):
    """The closing markdown cell: previous / next links, course map, and (for a
    core lesson) links to its companion variants."""
    m = meta(key)

    # A variant's footer just points home (and to its sibling, if any).
    if m["variant"] != "core":
        core = _core_of(m)
        parts = [f"**↩ Back to the lesson:** [{_short(core)}]({core['file']})",
                 "[↑ Course map](../rendered/index.html)"]
        for sib in _variants_of(core):
            if sib["id"] != m["id"]:
                parts.append(f"**Also:** [{_VARIANT_TAG[sib['variant']].capitalize()}]({sib['file']})")
        return "---\n\n" + "  ·  ".join(parts)

    # A core lesson: prev / map / next across the spine, then variant links.
    i = _SPINE.index(m)
    parts = []
    if i > 0:
        p = _SPINE[i - 1]
        parts.append(f"**← Previous:** [{_short(p)}]({p['file']})")
    parts.append("[↑ Course map](../rendered/index.html)")
    if i < len(_SPINE) - 1:
        n = _SPINE[i + 1]
        parts.append(f"**Next →:** [{_short(n)}]({n['file']})")
    nav = "---\n\n" + "  ·  ".join(parts)

    extras = _variants_of(m)
    if extras:
        links = "  ·  ".join(
            f"[{_VARIANT_TAG[v['variant']].capitalize()} →]({v['file']})" for v in extras)
        nav += f"\n\n**Same idea, another way:**  {links}"
    return nav


def _short(m):
    if m.get("orientation"):
        return "Start here: How these notebooks work"
    if m.get("flagship"):
        return "Flagship — Confidence intervals"
    return f"Lesson {_lesson_pos(m)}: {m['title']}"
