"""
Case vs. Control Gene Correlation Heatmaps
========================================================

Computes Pearson correlation matrices for a panel of genes, separately
within the case group and within the control group, and renders each as an
annotated heatmap. Comparing the two heatmaps highlights co-expression
patterns that differ between groups (e.g. a gene module that is tightly
co-regulated only in cases).

Expected input
--------------
  * An expression matrix CSV (probes/genes x samples), with the probe or
    gene identifier as the index column.
  * An annotation CSV with a sample identifier column and a group column
    indicating case vs. control status.

Usage
-----
    python correlation_heatmap.py \\
        data/processed/dataset_1_expression_matrix.csv data/processed/dataset_1_annotation.csv \\
        --probe-map GENE_A=probe_1 GENE_B=probe_2 \\
        --group-col group --case-label case --control-label control \\
        --output-dir results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_expression_subset(expr_path: Path, probe_to_gene: dict[str, str]) -> pd.DataFrame:
    """Load an expression matrix and subset/rename rows to a gene panel of interest."""
    df = pd.read_csv(expr_path, sep=None, engine="python")
    id_col = df.columns[0]
    df.columns = df.columns.str.replace('"', "", regex=False)
    df[id_col] = df[id_col].astype(str).str.replace('"', "", regex=False)

    df = df.set_index(id_col)
    wanted = set(probe_to_gene.keys())
    df = df.loc[df.index.isin(wanted)].rename(index=probe_to_gene)
    return df


def load_group_samples(
    annotation_path: Path, group_col: str, case_label: str, control_label: str, sample_col: str = "sample_id"
) -> tuple[list[str], list[str]]:
    """Return the (case_samples, control_samples) sample ID lists from an annotation file."""
    ann = pd.read_csv(annotation_path, sep=None, engine="python")
    ann.columns = ann.columns.str.strip()
    ann[sample_col] = ann[sample_col].astype(str).str.replace('"', "", regex=False)

    case_samples = ann.loc[ann[group_col] == case_label, sample_col].tolist()
    control_samples = ann.loc[ann[group_col] == control_label, sample_col].tolist()
    return case_samples, control_samples


def plot_correlation_heatmap(corr: pd.DataFrame, title: str, output_path: Path, dpi: int = 300) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="vlag",
        square=True, cbar_kws={"label": "Pearson r"},
    )
    plt.title(title, fontweight="bold", pad=12)
    plt.xticks(rotation=90, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _parse_probe_map(pairs: list[str]) -> dict[str, str]:
    probe_to_gene = {}
    for pair in pairs:
        gene, _, probe = pair.partition("=")
        probe_to_gene[probe] = gene
    return probe_to_gene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("expression_matrix", type=Path)
    parser.add_argument("annotation", type=Path)
    parser.add_argument(
        "--probe-map", nargs="+", required=True, metavar="GENE=PROBE_ID",
        help="Gene symbol to probe/feature ID mappings defining the panel to correlate.",
    )
    parser.add_argument("--group-col", default="group")
    parser.add_argument("--case-label", default="case")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    probe_to_gene = _parse_probe_map(args.probe_map)
    expr = load_expression_subset(args.expression_matrix, probe_to_gene)
    case_samples, control_samples = load_group_samples(
        args.annotation, args.group_col, args.case_label, args.control_label, args.sample_col
    )

    case_samples = [s for s in case_samples if s in expr.columns]
    control_samples = [s for s in control_samples if s in expr.columns]

    corr_case = expr[case_samples].T.corr(method="pearson")
    corr_control = expr[control_samples].T.corr(method="pearson")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_correlation_heatmap(corr_control, "Control", args.output_dir / "control_pearson_heatmap.png")
    plot_correlation_heatmap(corr_case, "Case", args.output_dir / "case_pearson_heatmap.png")
    print(f"Saved heatmaps -> {args.output_dir}")


if __name__ == "__main__":
    main()
