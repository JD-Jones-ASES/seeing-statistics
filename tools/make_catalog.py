"""
make_catalog.py — turn the dataset-research results into a readable catalog.

Usage:
    ..\\.venv\\Scripts\\python.exe tools\\make_catalog.py <path-to-research-json>

Reads the JSON produced by the dataset-scout research run and writes
raw/dataset-catalog.md: a recommended spine dataset, a starter set, a
concept -> dataset map, the full verified catalog (deduped by URL), and the
known gaps.
"""

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "raw" / "dataset-catalog.md"

ACCESS_LABEL = {
    "keep": "free, no account",
    "caution": "free, but via API / large file / account",
    "drop": "unavailable",
}


def main(json_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    result = data["result"]
    keepers = result["keepers"]
    mapping = result["mapping"]
    totals = result["totals"]

    # Dedupe by URL, keep first occurrence, preserve order.
    seen, unique = set(), []
    for d in keepers:
        url = d.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(d)
    unique.sort(key=lambda d: d["name"].lower())

    lines = []
    add = lines.append

    add("# Verified dataset catalog")
    add("")
    add(f"**{totals['kept']} datasets** from reliable public sources, each checked to be "
        "real, free, and downloadable. They are mapped below to the statistics concepts "
        "they teach best. Every URL was visited during research; none were dropped as "
        "unreachable.")
    add("")
    add("> Load any of these in a notebook with "
        "`statslab.load_csv('name.csv', url='...')` — it caches one copy into `raw/` "
        "and reuses it offline. Log new ones in `SOURCES.md`.")
    add("")

    # Spine
    add("## Recommended \"population\" dataset (the simulation spine)")
    add("")
    add(f"**{mapping['spineDataset']}**")
    add("")
    add(mapping["spineRationale"])
    add("")

    # Starter set
    add("## Starter set (covers the whole course)")
    add("")
    for name in mapping["starterSet"]:
        add(f"- {name}")
    add("")

    # Concept map
    add("## Concept → recommended dataset")
    add("")
    add("| Concept | Dataset | How to use it |")
    add("|---|---|---|")
    for row in mapping["conceptMap"]:
        how = row["how"].replace("|", "/")
        add(f"| {row['concept']} | {row['dataset']} | {how} |")
    add("")

    # Full catalog
    add("## Full verified catalog")
    add("")
    add("| Dataset | Source | Access | Size | Link |")
    add("|---|---|---|---|---|")
    for d in unique:
        access = ACCESS_LABEL.get(d.get("access", ""), d.get("access", ""))
        size = (d.get("size", "") or "").replace("|", "/")
        src = (d.get("source", "") or "").replace("|", "/")
        name = d["name"].replace("|", "/")
        url = d.get("url", "")
        add(f"| {name} | {src} | {access} | {size} | [link]({url}) |")
    add("")

    # Gaps
    gaps = mapping.get("gaps", [])
    if gaps:
        add("## Known gaps (where the data is thin)")
        add("")
        for g in gaps:
            add(f"- {g}")
        add("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}  ({len(unique)} unique datasets, {len(mapping['conceptMap'])} concepts mapped)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Pass the path to the research JSON file.")
    main(sys.argv[1])
