# Dataset sources (provenance log)

**Every dataset used in this course is recorded here: what it is, where it came
from, its license, and when it was downloaded.** This is validation
infrastructure, not paperwork — it keeps the course honest and lets anyone trace
a chart back to its raw source. Files in this `raw/` folder are treated as
**read-only originals**: we download a copy once and never edit it.

| Dataset (file) | Source & organization | Direct URL | License / terms | Downloaded | Used in |
|---|---|---|---|---|---|
| `penguins.csv` | Palmer Station LTER / Dr. Kristen Gorman; packaged by Allison Horst (`palmerpenguins`) | https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv | CC0 1.0 (public domain) | 2026-06-08 | Lesson 1 — Describing data; Lesson 3 — Relationships & regression (Simpson's paradox); Lesson 5 *data variant* — conditional probability & independence; Lesson 13 *data variant* — hypothesis testing (species masses); Lesson 16 — ANOVA; Lesson 17 *data variant* — chi-square (species × island); Lesson 19 *data variant* — multiple regression (mass ~ flipper + species) |
| `faithful.csv` | Old Faithful geyser eruptions; W. Härdle (1991); base-R `datasets`, mirrored by Vincent Arel-Bundock (Rdatasets) | https://vincentarelbundock.github.io/Rdatasets/csv/datasets/faithful.csv | Public domain / GPL-2 (R datasets) | 2026-06-08 | Lesson 2 — Distributions |
| `loans.csv` | Lending Club peer-to-peer loans; curated by OpenIntro (`loans_full_schema`) | https://www.openintro.org/data/csv/loans_full_schema.csv | Free for educational use (OpenIntro) | 2026-06-08 | Lesson 2 — Distributions; Lesson 10 — CLT |
| `titanic.csv` | Titanic passenger manifest; Vanderbilt Biostatistics (F. Harrell) / historical records; Stanford CS109 mirror | https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv | Public domain (historical record) | 2026-06-08 | Lesson 5 — Probability; Lesson 7 *data variant* — random variables (relatives aboard); Lesson 8 *data variant* — binomial (survival); Lesson 14 *data variant* — inference for proportions (survival by sex) |
| `births14.csv` | US CDC/NCHS Natality Detail File 2014; curated by OpenIntro (`births14`) | https://www.openintro.org/data/csv/births14.csv | Free for educational use (OpenIntro); public source data | 2026-06-08 | Lesson 7 — Random variables; Lesson 8 — Binomial & Poisson |
| `earthquakes_2024_06.csv` | M4.5+ global earthquakes, June 2024; USGS Earthquake Hazards Program (ANSS ComCat), FDSN query | https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2024-06-01&endtime=2024-07-01&minmagnitude=4.5&orderby=time-asc | US Government work, public domain | 2026-06-08 | Lesson 8 — Binomial & Poisson; Lesson 8 *data variant* — Poisson on rarer (M5+) quakes per day |
| `galton.csv` | Galton family heights; F. Galton (1886); `HistData` package, mirrored by Vincent Arel-Bundock (Rdatasets) | https://vincentarelbundock.github.io/Rdatasets/csv/HistData/GaltonFamilies.csv | Public domain (historical) / GPL (HistData) | 2026-06-08 | Lesson 9 — Normal distribution |
| `ames.csv` | Ames, Iowa City Assessor (public record); De Cock (2011, *J. Statistics Education*); distributed by OpenIntro | https://www.openintro.org/data/csv/ames.csv | Public data, free for educational use (OpenIntro) | 2026-06-08 | Lesson 3 — Relationships & regression; Lesson 4 — Sampling & study design; Lesson 10 *data variant* — CLT (sale price); Lesson 11 — Estimation; Lesson 12 — Confidence intervals (flagship); Lesson 12 *data variant* — confidence intervals (sale price); Lessons 18–19 — regression inference |
| `ToothGrowth.csv` | Guinea-pig odontoblast length by vitamin-C supplement (OJ/VC) and dose; Crampton (1947); base-R `datasets`, mirrored by Vincent Arel-Bundock (Rdatasets) | https://vincentarelbundock.github.io/Rdatasets/csv/datasets/ToothGrowth.csv | Public domain / GPL-2 (R datasets) | 2026-06-09 | Lesson 15 — t-tests; Lesson 20 — Bootstrap & permutation |
| `swim.csv` | Swimmer velocities, wetsuit vs swimsuit (paired, n=12); De Lucas et al. (2000); curated by OpenIntro | https://www.openintro.org/data/csv/swim.csv | Free for educational use (OpenIntro) | 2026-06-09 | Lesson 15 — t-tests (paired); Lesson 20 *data variant* — bootstrap & permutation (paired diffs) |
| `iris.csv` | Fisher's irises (1936); Edgar Anderson measurements; base-R `datasets`, mirrored by Vincent Arel-Bundock (Rdatasets) | https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv | Public domain / GPL-2 (R datasets) | 2026-06-09 | Lesson 1 *data variant* — describing a clean dataset |
| `gapminder.csv` | Gapminder Foundation country-year panel (1952–2007); Jenny Bryan / kirenz GitHub mirror | https://raw.githubusercontent.com/kirenz/datasets/master/gapminder.csv | Free, no account (Gapminder data, CC-BY) | 2026-06-09 | Lesson 2 *data variant* — left-skew (life expectancy); Lesson 3 *data variant* — nonlinearity & log transform (GDP vs life expectancy); Lesson 18 *data variant* — regression inference (slope on log GDP) |
| `nhanes_men.csv` | NHANES 2015–2018 adult-male body measurements (height `BMXHT` in mm); CDC / NCHS, parsed by Penn State OPEN Design Lab | https://bpb-us-e1.wpmucdn.com/sites.psu.edu/dist/4/27975/files/2023/09/NHANES15-18_menAge20YearsAndOver.csv | US Government work, public domain (NHANES); CSV mirror by Penn State OPEN Design Lab | 2026-06-09 | Lesson 9 *data variant* — the empirical rule on ~5,000 heights; Lesson 11 *data variant* — estimation & standard error |
| `chickwts.csv` | Chick weights by feed supplement (6 feeds); agricultural experiment, base-R `datasets`, mirrored by Vincent Arel-Bundock (Rdatasets) | https://vincentarelbundock.github.io/Rdatasets/csv/datasets/chickwts.csv | Public domain / GPL-2 (R datasets) | 2026-06-09 | Lesson 15 *data variant* — two-sample t-test (two feeds); Lesson 16 *data variant* — ANOVA (six feeds) |
| `textbooks.csv` | UCLA bookstore vs Amazon new-textbook prices (paired, 73 books); curated by OpenIntro | https://www.openintro.org/data/csv/textbooks.csv | Free for educational use (OpenIntro) | 2026-06-09 | Lesson 15 *data variant* — paired t-test (campus vs online price) |

---

**The full menu:** 50+ additional verified, free, no-account datasets — each mapped
to the statistics concepts it teaches best — are catalogued in
[`dataset-catalog.md`](dataset-catalog.md).

**How a dataset gets here:** `lib/statslab.load_csv(filename, url=...)` downloads
one cached copy into this folder the first time a lesson runs, then reuses it
offline forever after. Add a row above (source, license, date) whenever a new
dataset is introduced.
