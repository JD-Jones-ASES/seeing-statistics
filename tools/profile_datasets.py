r"""
profile_datasets.py — download & profile candidate datasets for new lessons.

Confirms each URL is reachable (via statslab.load_csv, which caches into the
read-only raw/ folder with the right User-Agent), then prints the REAL columns,
dtypes, sizes, and key summary statistics. We author lessons against this output
so no lesson ever references a guessed column name or a dead link.

Run:  ..\.venv\Scripts\python.exe tools\profile_datasets.py
"""
from __future__ import annotations
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import statslab as sl
import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)

# (filename in raw/, download url, key columns to summarize)
CANDIDATES = [
    ("faithful.csv",
     "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/faithful.csv",
     ["eruptions", "waiting"]),
    ("galton.csv",
     "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/GaltonFamilies.csv",
     ["father", "mother", "midparentHeight", "childHeight"]),
    ("loans.csv",
     "https://www.openintro.org/data/csv/loans_full_schema.csv",
     ["annual_income", "interest_rate", "loan_amount"]),
    ("titanic.csv",
     "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv",
     []),
    ("births14.csv",
     "https://www.openintro.org/data/csv/births14.csv",
     ["weight", "weeks", "visits", "mage"]),
    ("earthquakes_2024_06.csv",
     "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2024-06-01&endtime=2024-07-01&minmagnitude=4.5&orderby=time-asc",
     ["mag", "depth"]),
    ("nhanes_men.csv",
     "https://bpb-us-e1.wpmucdn.com/sites.psu.edu/dist/4/27975/files/2023/09/NHANES15-18_menAge20YearsAndOver.csv",
     []),
]


def summarize(name, df, keys):
    print("=" * 78)
    print(f"{name}   shape = {df.shape[0]:,} rows x {df.shape[1]} cols")
    print("-" * 78)
    print("columns:", list(df.columns))
    print("-" * 78)
    print("dtypes / non-null:")
    print(df.dtypes.to_string())
    print("-" * 78)
    print("head(3):")
    print(df.head(3).to_string())
    for k in keys:
        if k in df.columns:
            s = pd.to_numeric(df[k], errors="coerce").dropna()
            if len(s):
                print("-" * 78)
                print(f"[{k}] n={len(s):,} miss={df[k].isna().sum()} "
                      f"mean={s.mean():.3f} median={s.median():.3f} "
                      f"std={s.std(ddof=1):.3f} min={s.min():.3f} max={s.max():.3f} "
                      f"skew={s.skew():.3f}")
    print()


def main():
    for name, url, keys in CANDIDATES:
        try:
            df = sl.load_csv(name, url=url)
            summarize(name, df, keys)
        except Exception as e:
            print("=" * 78)
            print(f"FAILED: {name}\n  url={url}\n  error={type(e).__name__}: {e}\n")

    # Extra: Titanic conditional probabilities (Lesson 3 needs real numbers)
    try:
        t = sl.load_csv("titanic.csv")
        print("#" * 78)
        print("TITANIC cross-tabs for Lesson 3")
        sex_col = next((c for c in t.columns if c.lower() == "sex"), None)
        surv_col = next((c for c in t.columns if "surviv" in c.lower()), None)
        cls_col = next((c for c in t.columns if "class" in c.lower() or c.lower() == "pclass"), None)
        print("detected columns -> sex:", sex_col, "survived:", surv_col, "class:", cls_col)
        if surv_col:
            print(f"P(survived) = {t[surv_col].mean():.4f}")
        if sex_col and surv_col:
            print("P(survived | sex):")
            print(t.groupby(sex_col)[surv_col].mean().to_string())
        if cls_col and surv_col:
            print("P(survived | class):")
            print(t.groupby(cls_col)[surv_col].mean().to_string())
    except Exception as e:
        print("titanic cross-tab failed:", e)

    # Extra: earthquakes daily counts (Lesson 5 Poisson suitability)
    try:
        q = sl.load_csv("earthquakes_2024_06.csv")
        print("#" * 78)
        print("EARTHQUAKES daily counts for Lesson 5 (Poisson)")
        when = pd.to_datetime(q["time"]).dt.date
        daily = when.value_counts().sort_index()
        full = pd.Series(0, index=pd.date_range("2024-06-01", "2024-06-30").date)
        full.update(daily)
        print(f"days={len(full)} total_events={int(full.sum())} "
              f"mean/day={full.mean():.3f} var/day={full.var(ddof=1):.3f} "
              f"max/day={int(full.max())}  (Poisson wants mean ~= var)")
        print("count-of-counts:", dict(full.value_counts().sort_index()))
    except Exception as e:
        print("earthquake daily-count failed:", e)

    # Extra: births14 low-birthweight proportion (Lesson 5 binomial / later CI-for-proportion)
    try:
        b = sl.load_csv("births14.csv")
        print("#" * 78)
        print("BIRTHS14 detail for Lessons 4/5")
        lbw = next((c for c in b.columns if "lowbirth" in c.lower() or c.lower() == "lowbirthweight"), None)
        print("columns:", list(b.columns))
        if lbw:
            print(f"low-birthweight column = {lbw}; value_counts:")
            print(b[lbw].value_counts(dropna=False).to_string())
        if "visits" in b.columns:
            v = pd.to_numeric(b["visits"], errors="coerce").dropna()
            print(f"visits: mean={v.mean():.3f} var={v.var(ddof=1):.3f} "
                  f"min={v.min():.0f} max={v.max():.0f}")
            print("visits value_counts (head):", dict(v.astype(int).value_counts().sort_index().head(15)))
    except Exception as e:
        print("births detail failed:", e)


if __name__ == "__main__":
    main()
