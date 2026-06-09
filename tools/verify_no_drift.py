r"""verify_no_drift.py — confirm a re-render changed only visuals, not statistics.

Every lesson uses fixed random seeds, so a clean re-execution is deterministic:
the *numbers* (printed text and text/plain table reprs) must be byte-identical to
the committed baseline. Only the embedded chart images may differ. This script
compares the TEXT outputs of every executed notebook in the working tree against a
git baseline (default: HEAD), normalising away genuinely-volatile bits (memory
addresses, bare matplotlib/Figure object reprs). It prints the first real
divergence per lesson and exits non-zero if any numbers moved.

    ..\.venv\Scripts\python.exe tools\verify_no_drift.py            # compare vs HEAD
    ..\.venv\Scripts\python.exe tools\verify_no_drift.py <git-ref>  # vs another commit

Run from the statistics-course/ git root. Datasets in raw/ are only ever read.
"""
import re
import sys
import subprocess
import pathlib

import nbformat

ROOT = pathlib.Path(__file__).resolve().parent.parent
LESSONS = ROOT / "lessons"

ADDR = re.compile(r"0x[0-9A-Fa-f]+")
# Lines that are just a stray matplotlib/Figure repr (object identity, not data).
VOLATILE_LINE = re.compile(
    r"^\s*(\[?<[^>]*(matplotlib|Figure|Axes|Line2D|Patch|Collection)[^>]*>\]?|"
    r"<Figure size[^>]*>|Text\(.*\))\s*$"
)

# Tokens that legitimately vary run-to-run and are NOT statistics: wall-clock
# timings (benchmark cells) and the fit timestamp statsmodels stamps into its
# summary header. We blank just the token, in place, so real numbers on the same
# line (e.g. Log-Likelihood) and overall line alignment are preserved.
TOKEN_SUBS = [
    (re.compile(r"\b\d[\d,]*\.?\d*\s*(ms|µs|us|ns)\b", re.I), "<t>"),     # 678.7 ms
    (re.compile(r"\b\d+(\.\d+)?x faster\b"), "<n>x faster"),               # 7x faster
    (re.compile(r"(Time:\s*)\d{1,2}:\d{2}:\d{2}"), r"\1<ts>"),             # statsmodels Time:
    (re.compile(r"(Date:\s*)\w{3},\s*\d{1,2}\s*\w{3}\s*\d{4}"), r"\1<d>"), # statsmodels Date:
    (re.compile(r"Wall time:.*$"), "Wall time: <t>"),
    (re.compile(r"CPU times:.*$"), "CPU times: <t>"),
    (re.compile(r"per loop \(mean ± .*\)"), "per loop (<timeit>)"),
    (re.compile(r"ipykernel_\d+"), "ipykernel_<pid>"),          # temp-kernel PID in warning paths
    (re.compile(r"\b\d{6,}\.py\b"), "<cell>.py"),               # temp per-cell module filename
]


def _scrub(line):
    line = ADDR.sub("0xADDR", line.rstrip())
    for rx, repl in TOKEN_SUBS:
        line = rx.sub(repl, line)
    # collapse internal whitespace so column-padding that shifts with a number's
    # digit-width (e.g. "108.1 ms" vs " 97.8 ms") doesn't read as a difference.
    return re.sub(r"[ \t]+", " ", line).strip()


def text_outputs(nb):
    """Ordered list of a notebook's textual outputs (images excluded)."""
    chunks = []
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            kind = out.get("output_type")
            if kind == "stream":
                chunks.append(out.get("text", ""))
            elif kind in ("execute_result", "display_data"):
                txt = out.get("data", {}).get("text/plain")
                if txt:
                    chunks.append(txt)
            elif kind == "error":
                chunks.append("ERROR: " + "\n".join(out.get("traceback", [])))
    return chunks


def normalize(chunks):
    lines = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if VOLATILE_LINE.match(line):
                continue
            scrubbed = _scrub(line)
            if scrubbed == "":      # blank lines carry no statistics; drop so counts can't misalign
                continue
            lines.append(scrubbed)
    return lines


def baseline_nb(ref, relpath):
    """Read lessons/<file> from a git ref; None if it didn't exist there."""
    try:
        blob = subprocess.run(
            ["git", "show", f"{ref}:{relpath}"],
            cwd=ROOT, capture_output=True, check=True,
        ).stdout.decode("utf-8")
    except subprocess.CalledProcessError:
        return None
    return nbformat.reads(blob, as_version=4)


def main(argv):
    ref = argv[0] if argv else "HEAD"
    ok, drifted, new = 0, [], []
    for nb_path in sorted(LESSONS.glob("*.ipynb")):
        rel = f"lessons/{nb_path.name}"
        cur = normalize(text_outputs(nbformat.read(nb_path, as_version=4)))
        base_nb = baseline_nb(ref, rel)
        if base_nb is None:
            new.append(nb_path.name)
            continue
        base = normalize(text_outputs(base_nb))
        if cur == base:
            ok += 1
            continue
        # find first differing line
        first = next((i for i in range(max(len(cur), len(base)))
                      if (cur[i] if i < len(cur) else None)
                      != (base[i] if i < len(base) else None)), 0)
        drifted.append((nb_path.name, first,
                        base[first] if first < len(base) else "<missing>",
                        cur[first] if first < len(cur) else "<missing>"))

    print(f"clean (numbers identical): {ok}")
    if new:
        print(f"new (no baseline, skipped): {len(new)} -> {', '.join(new)}")
    if drifted:
        print(f"\nDRIFT in {len(drifted)} lesson(s) — numbers changed:")
        for name, i, b, c in drifted:
            print(f"  {name}  (first diff at text-line {i})")
            print(f"    baseline: {b!r}")
            print(f"    current : {c!r}")
        return 1
    print("\nNo statistical drift: every number matches the baseline. Only visuals changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
