"""
Cross-Condition Drug/Indication Overlap Analysis
========================================================

Given a drug-gene (or drug-target) interaction table for each of several
conditions (e.g. one CSV per case-control study, each listing candidate
drugs and their indications for that condition's dysregulated genes), this
script finds which drugs and which indications are shared across every
pairwise combination of conditions and across all conditions at once.

It also classifies each shared indication as "Activating" or "Inhibiting"
based on keyword matching against its free-text description (e.g. "agonist",
"inducer" -> Activating; "antagonist", "inhibitor" -> Inhibiting), which is
useful for distinguishing drugs expected to reverse vs. reinforce a
regulation direction.

Expected input
--------------
One CSV file per condition, each containing at least a ``drug`` column and
an ``indication`` column (free-text description of the drug's mechanism of
action / indication, e.g. as exported from a drug-gene interaction database).

Usage
-----
    python condition_overlap_analysis.py \\
        --input condition_1=data/processed/condition_1_interactions.csv \\
        --input condition_2=data/processed/condition_2_interactions.csv \\
        --input condition_3=data/processed/condition_3_interactions.csv \\
        --column indication \\
        --output-dir results/
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

ACTIVATING_KEYWORDS = [
    "agonist", "activator", "stimulating", "stimulator", "stimulate", "enhancer",
    "inducing", "induced", "upregulator", "upregulating", "upregulation",
    "positive modulator", "co-agonist", "positive allosteric modulator",
    "sensitizer", "reactivator", "openers", "releasing agent",
]

INHIBITING_KEYWORDS = [
    "antagonist", "inhibitor", "suppressor", "blocking", "blocker", "suppressing",
    "suppress", "decreasing", "downregulator", "downregulating", "downregulation",
    "antagonist activity", "reducing", "negative modulator", "allosteric modulator",
    "negative allosteric modulator", "negative regulator", "interfering",
    "decreasing agent", "closed", "disruptor", "silencer",
]


def categorize_indication(indication: str) -> str:
    """Classify a free-text indication as Activating, Inhibiting, or N/A.

    Inhibiting keywords take priority over activating keywords when both
    happen to match (mirrors the original keyword-scan order).
    """
    text = str(indication).lower()
    if any(kw in text for kw in INHIBITING_KEYWORDS):
        return "Inhibiting"
    if any(kw in text for kw in ACTIVATING_KEYWORDS):
        return "Activating"
    return "N/A"


def load_condition_tables(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path) for name, path in paths.items()}


def _values(df: pd.DataFrame, column: str) -> set[str]:
    return set(df[column].dropna().astype(str).str.strip())


def summarize_overlap(tables: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """For each combination of 2+ conditions, list the values shared between them.

    Returns a long-format DataFrame with columns:
      ``value``, ``shared_between`` (conditions joined by '&'), ``n_conditions``
      and, when ``column == "indication"``, a ``category`` column
      (Activating / Inhibiting / N/A).
    """
    names = list(tables.keys())
    value_sets = {name: _values(df, column) for name, df in tables.items()}

    rows = []
    for r in range(2, len(names) + 1):
        for combo in combinations(names, r):
            shared = set.intersection(*(value_sets[name] for name in combo))
            for value in sorted(shared):
                row = {
                    "value": value,
                    "shared_between": " & ".join(combo),
                    "n_conditions": r,
                }
                if column == "indication":
                    row["category"] = categorize_indication(value)
                rows.append(row)

    return pd.DataFrame(rows)


def matched_rows(tables: dict[str, pd.DataFrame], column: str, overlap: pd.DataFrame) -> pd.DataFrame:
    """Return the full source rows for every shared value, tagged by condition."""
    frames = []
    for name, df in tables.items():
        shared_values = set(overlap.loc[overlap["shared_between"].str.contains(name), "value"])
        if not shared_values:
            continue
        subset = df[df[column].astype(str).str.strip().isin(shared_values)].copy()
        subset.insert(0, "condition", name)
        frames.append(subset)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _parse_input_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH, e.g. condition_1=data/processed/condition_1.csv")
    name, _, path = value.partition("=")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        action="append",
        type=_parse_input_arg,
        required=True,
        metavar="NAME=PATH",
        help="A condition name and its interaction-table CSV path. Repeat for each condition (at least 2).",
    )
    parser.add_argument(
        "--column",
        default="indication",
        help="Column to compute overlaps on, e.g. 'indication' or 'drug' (default: indication).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Output directory (default: results).")
    args = parser.parse_args()

    if len(args.input) < 2:
        parser.error("At least two --input NAME=PATH entries are required.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_condition_tables(dict(args.input))

    overlap = summarize_overlap(tables, args.column)
    overlap_path = args.output_dir / f"shared_{args.column}_summary.csv"
    overlap.to_csv(overlap_path, index=False)
    print(f"Saved {len(overlap)} shared '{args.column}' entries -> {overlap_path}")

    rows = matched_rows(tables, args.column, overlap)
    rows_path = args.output_dir / f"shared_{args.column}_matched_rows.csv"
    rows.to_csv(rows_path, index=False)
    print(f"Saved matched source rows -> {rows_path}")


if __name__ == "__main__":
    main()
