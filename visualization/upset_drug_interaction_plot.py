"""
UpSet Plot of Shared Drug-Interaction Scores Across Conditions
================================================================

Visualizes drugs shared between combinations of conditions as an UpSet plot,
with bar height weighted by (summed or averaged) drug-gene interaction
score rather than a plain intersection count. Only 2-way and higher
intersections are shown (i.e. drugs unique to a single condition are
excluded), matching the "candidate repurposing drug" framing of highlighting
drugs relevant to multiple conditions at once.

Expected input
--------------
One CSV file per condition *combination* (pairwise and, for 3+ conditions,
the all-conditions intersection), each containing at least ``drug`` and
``interaction score`` columns -- i.e. a precomputed intersection table
(for example, the output of an upstream overlap step such as
``condition_overlap_analysis.py`` joined with interaction scores).

Usage
-----
    python upset_drug_interaction_plot.py \\
        --input "condition_1 & condition_2"=data/processed/common_1_2.csv \\
        --input "condition_2 & condition_3"=data/processed/common_2_3.csv \\
        --input "condition_1 & condition_3"=data/processed/common_1_3.csv \\
        --input "condition_1 & condition_2 & condition_3"=data/processed/common_1_2_3.csv \\
        --aggregate sum \\
        --output results/upset_drug_interactions.png
"""

from __future__ import annotations

import argparse
from itertools import combinations, chain
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_memberships

DEFAULT_PALETTE = [
    "#add8e6", "#dab6f3", "#c0e8c0", "#ffa07a", "#ffe08a", "#f4a6c6", "#b0c4de",
]


def _parse_combo(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError('Expected "COND_A & COND_B"=PATH')
    label, _, path = value.partition("=")
    return label.strip(), Path(path)


def build_color_map(condition_names: list[str]) -> dict[frozenset, str]:
    """Assign a default color to every individual condition and every 2+ combination."""
    color_map: dict[frozenset, str] = {}
    combos: list[tuple[str, ...]] = []
    for r in range(1, len(condition_names) + 1):
        combos.extend(combinations(condition_names, r))
    for combo, color in zip(combos, chain.from_iterable([DEFAULT_PALETTE] * 10)):
        color_map[frozenset(combo)] = color
    return color_map


def load_drug_scores(inputs: dict[str, Path]) -> tuple[dict[str, set], dict[str, float]]:
    """Read each combination file and collapse to per-drug (member-set, best score)."""
    drug_to_sets: dict[str, set] = {}
    drug_to_score: dict[str, float] = {}

    for label, path in inputs.items():
        conditions = tuple(sorted(c.strip() for c in label.split("&")))
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            drug = str(row["drug"]).strip()
            score = float(row["interaction score"])
            if drug not in drug_to_sets:
                drug_to_sets[drug] = set(conditions)
                drug_to_score[drug] = score
            else:
                drug_to_sets[drug].update(conditions)
                drug_to_score[drug] = max(drug_to_score[drug], score)

    return drug_to_sets, drug_to_score


def build_upset_data(
    drug_to_sets: dict[str, set], drug_to_score: dict[str, float], condition_names: list[str], aggregate: str
) -> pd.Series:
    """Build the UpSet-compatible series, restricted to 2+ way intersections."""
    valid_combinations = [
        tuple(sorted(combo)) for r in range(2, len(condition_names) + 1) for combo in combinations(condition_names, r)
    ]

    existing = {tuple(sorted(v)) for v in drug_to_sets.values()}
    for combo in valid_combinations:
        if combo not in existing:
            drug_to_sets[f"__dummy_{'_'.join(combo)}"] = set(combo)
            drug_to_score[f"__dummy_{'_'.join(combo)}"] = 0.0

    memberships = [tuple(sorted(v)) for v in drug_to_sets.values()]
    data = [drug_to_score[d] for d in drug_to_sets]

    series = from_memberships(memberships, data=data)
    agg_func = {"sum": "sum", "mean": "mean"}[aggregate]
    series = series.groupby(level=list(range(series.index.nlevels))).agg(agg_func)
    series = series[[len([i for i in idx if i]) >= 2 for idx in series.index]]
    return series


def plot_upset(series: pd.Series, color_map: dict[frozenset, str], output_path: Path, ylabel: str) -> None:
    n_conditions = series.index.nlevels

    fig = plt.figure(figsize=(25, 6))
    upset = UpSet(series, show_counts=False, show_percentages=False, min_subset_size=0)
    upset.plot(fig=fig)

    for ax in fig.axes:
        ax.grid(False)

        if ax.get_ylabel() == "Intersection size":
            # Top bars: one per condition combination. Kept a neutral color
            # with the aggregate score annotated above each bar.
            ax.set_ylabel(ylabel, fontsize=16)
            for bar in ax.patches:
                bar.set_color("#000000")
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, height + max(height * 0.02, 0.01),
                        f"{height:.2f}", ha="center", va="bottom", fontsize=12, rotation=90,
                    )
            continue

        # The per-condition "totals" bar chart (left side panel) is the only
        # other axis with exactly one bar per condition; it is identified by
        # its bar count and its y-tick labels (which give the condition
        # names directly), since neither its ylabel nor its position is
        # stable across upsetplot versions.
        tick_labels = [t.get_text() for t in ax.get_yticklabels()]
        if len(ax.patches) == n_conditions and any(tick_labels):
            for bar, label in zip(ax.patches, tick_labels):
                color = color_map.get(frozenset({label}), "#a9a9a9")
                bar.set_color(color)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", action="append", type=_parse_combo, required=True, metavar='"COND_A & COND_B"=PATH',
        help="A condition-combination label and its intersection-table CSV. Repeat for every 2+ combination.",
    )
    parser.add_argument("--aggregate", choices=["sum", "mean"], default="sum", help="Score aggregation (default: sum).")
    parser.add_argument("--output", type=Path, default=Path("results/upset_drug_interactions.png"))
    args = parser.parse_args()

    inputs = dict(args.input)
    condition_names = sorted({c.strip() for label in inputs for c in label.split("&")})

    drug_to_sets, drug_to_score = load_drug_scores(inputs)
    series = build_upset_data(drug_to_sets, drug_to_score, condition_names, args.aggregate)
    color_map = build_color_map(condition_names)

    ylabel = "Total Interaction Score" if args.aggregate == "sum" else "Mean Interaction Score"
    plot_upset(series, color_map, args.output, ylabel)
    print(f"Saved UpSet plot -> {args.output}")


if __name__ == "__main__":
    main()
