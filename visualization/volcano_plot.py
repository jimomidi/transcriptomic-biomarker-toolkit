"""
Volcano Plot for Differential Expression Results
========================================================

Classifies genes as up-/down-regulated based on a p-value and log2
fold-change threshold, then renders a volcano plot. Two styles are provided:

  * a standard three-color volcano plot (Not significant / Upregulated /
    Downregulated), and
  * a highlighted variant that greys out all genes except a supplied panel
    of genes of interest, each drawn in its own color -- useful for
    spotlighting a predefined biomarker panel against the full background
    distribution.

Both styles also write out the table of significant genes (Up/Down) as CSV.

Expected input
--------------
A CSV of per-gene differential expression statistics with at minimum a gene
identifier column, a ``logFC`` column, and a p-value column (e.g.
``P.Value`` or ``adj.P.Val``, as produced by a standard differential
expression tool such as limma or GEO2R). This script does not compute
differential expression itself -- it consumes a precomputed statistics
table.

Usage
-----
    python volcano_plot.py data/processed/dataset_1_de_results.csv \\
        --gene-col gene --pvalue-col P.Value --logfc-threshold 0.3 --pvalue-threshold 0.05 \\
        --output results/dataset_1_volcano.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

DEFAULT_LOGFC_THRESHOLD = 0.3
DEFAULT_PVALUE_THRESHOLD = 0.05

HIGHLIGHT_PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33",
    "#A65628", "#F781BF", "#999999", "#1B9E77", "#39FF14", "#7570B3",
]


def classify_direction(
    df: pd.DataFrame,
    logfc_col: str = "logFC",
    pvalue_col: str = "P.Value",
    logfc_threshold: float = DEFAULT_LOGFC_THRESHOLD,
    pvalue_threshold: float = DEFAULT_PVALUE_THRESHOLD,
) -> pd.DataFrame:
    """Add ``Significant`` and ``Direction`` columns to a DE results table."""
    df = df.copy()
    significant = (df[pvalue_col] < pvalue_threshold) & (df[logfc_col].abs() > logfc_threshold)
    df["Significant"] = significant
    df["Direction"] = np.where(
        significant & (df[logfc_col] > logfc_threshold),
        "Up",
        np.where(significant & (df[logfc_col] < -logfc_threshold), "Down", "NotSig"),
    )
    return df


def plot_volcano(
    df: pd.DataFrame,
    output_path: Path,
    logfc_col: str = "logFC",
    pvalue_col: str = "P.Value",
    pvalue_threshold: float = DEFAULT_PVALUE_THRESHOLD,
    logfc_threshold: float = DEFAULT_LOGFC_THRESHOLD,
    title: str = "Volcano Plot",
) -> None:
    """Render a standard three-color (Not significant / Up / Down) volcano plot."""
    fig, ax = plt.subplots(figsize=(7, 7))

    colors = {"NotSig": "grey", "Up": "red", "Down": "blue"}
    labels = {"NotSig": "Not significant", "Up": "Upregulated", "Down": "Downregulated"}
    for direction, color in colors.items():
        subset = df[df["Direction"] == direction]
        ax.scatter(
            subset[logfc_col], -np.log10(subset[pvalue_col]),
            color=color, alpha=0.6, label=labels[direction],
        )

    ax.axvline(x=-logfc_threshold, color="blue", linestyle="--", lw=1)
    ax.axvline(x=logfc_threshold, color="red", linestyle="--", lw=1)
    ax.axhline(y=-np.log10(pvalue_threshold), color="green", linestyle="--", lw=1)

    ax.set_xlabel("log2 Fold Change", fontsize=14)
    ax.set_ylabel(f"-log10 {pvalue_col}", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_volcano_with_highlights(
    df: pd.DataFrame,
    gene_col: str,
    highlight_genes: dict[str, str],
    output_path: Path,
    logfc_col: str = "logFC",
    pvalue_col: str = "P.Value",
    pvalue_threshold: float = DEFAULT_PVALUE_THRESHOLD,
    logfc_threshold: float = DEFAULT_LOGFC_THRESHOLD,
    title: str = "",
    xlim: tuple[float, float] = (-1.0, 1.0),
) -> None:
    """Render a volcano plot that highlights a predefined gene panel.

    Background genes are drawn in translucent black; each gene in
    ``highlight_genes`` (mapping gene symbol -> the identifier used in
    ``gene_col``) is drawn in its own color and added to the legend.
    """
    fig, ax = plt.subplots(figsize=(5, 6))
    y_cut = -np.log10(pvalue_threshold)

    highlight_ids = set(highlight_genes.values())
    background = ~df[gene_col].isin(highlight_ids)
    ax.scatter(
        df.loc[background, logfc_col], -np.log10(df.loc[background, pvalue_col]),
        color="black", alpha=0.4, s=22,
    )

    ax.axvline(x=-logfc_threshold, color="black", linestyle="--", lw=1)
    ax.axvline(x=logfc_threshold, color="black", linestyle="--", lw=1)
    ax.axhline(y=y_cut, color="black", linestyle="--", lw=1)

    ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.add_patch(Polygon(
        [[logfc_threshold, y_cut], [xlim[1], y_cut], [xlim[1], ylim[1]], [logfc_threshold, ylim[1]]],
        facecolor="#FFB3B3", alpha=0.35, edgecolor=None, zorder=0,
    ))
    ax.add_patch(Polygon(
        [[xlim[0], y_cut], [-logfc_threshold, y_cut], [-logfc_threshold, ylim[1]], [xlim[0], ylim[1]]],
        facecolor="#B3E0FF", alpha=0.35, edgecolor=None, zorder=0,
    ))

    for idx, (gene_symbol, gene_id) in enumerate(highlight_genes.items()):
        subset = df[df[gene_col] == gene_id]
        if subset.empty:
            continue
        ax.scatter(
            subset[logfc_col], -np.log10(subset[pvalue_col]),
            color=HIGHLIGHT_PALETTE[idx % len(HIGHLIGHT_PALETTE)],
            s=110, edgecolor=None, label=gene_symbol, zorder=10,
        )

    ax.set_xlabel("log2 Fold Change", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel(f"-log10 {pvalue_col}", fontsize=18, fontweight="bold", labelpad=10)
    if title:
        ax.set_title(title, fontsize=20, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_significant_genes(df: pd.DataFrame, gene_col: str, logfc_col: str, pvalue_col: str, output_path: Path) -> pd.DataFrame:
    sig = df[df["Direction"].isin(["Up", "Down"])][[gene_col, logfc_col, pvalue_col, "Direction"]]
    sig.to_csv(output_path, index=False)
    return sig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("de_results", type=Path, help="CSV of differential expression statistics.")
    parser.add_argument("--gene-col", default="gene", help="Gene/probe identifier column name (default: gene).")
    parser.add_argument("--logfc-col", default="logFC", help="Log2 fold-change column name (default: logFC).")
    parser.add_argument(
        "--pvalue-col", default="P.Value", help="P-value column name, e.g. P.Value or adj.P.Val (default: P.Value)."
    )
    parser.add_argument("--logfc-threshold", type=float, default=DEFAULT_LOGFC_THRESHOLD)
    parser.add_argument("--pvalue-threshold", type=float, default=DEFAULT_PVALUE_THRESHOLD)
    parser.add_argument("--title", default="Volcano Plot")
    parser.add_argument("--output", type=Path, default=Path("results/volcano_plot.png"))
    parser.add_argument(
        "--highlight-genes", nargs="+", default=None, metavar="GENE=ID",
        help="Optional gene panel to highlight (e.g. GENE_A=probe_1 GENE_B=probe_2). "
             "When given, an additional highlighted-panel volcano plot is also rendered.",
    )
    parser.add_argument(
        "--highlight-xlim", type=float, nargs=2, default=(-1.0, 1.0), metavar=("MIN", "MAX"),
        help="X-axis limits for the highlighted-panel plot (default: -1.0 1.0).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.de_results)
    df = classify_direction(
        df,
        logfc_col=args.logfc_col,
        pvalue_col=args.pvalue_col,
        logfc_threshold=args.logfc_threshold,
        pvalue_threshold=args.pvalue_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_volcano(
        df, args.output,
        logfc_col=args.logfc_col, pvalue_col=args.pvalue_col,
        pvalue_threshold=args.pvalue_threshold, logfc_threshold=args.logfc_threshold,
        title=args.title,
    )
    print(f"Saved volcano plot -> {args.output}")

    if args.highlight_genes:
        highlight_genes = dict(pair.split("=", 1) for pair in args.highlight_genes)
        highlight_output = args.output.with_name(args.output.stem + "_highlighted.png")
        plot_volcano_with_highlights(
            df, args.gene_col, highlight_genes, highlight_output,
            logfc_col=args.logfc_col, pvalue_col=args.pvalue_col,
            pvalue_threshold=args.pvalue_threshold, logfc_threshold=args.logfc_threshold,
            title=args.title, xlim=tuple(args.highlight_xlim),
        )
        print(f"Saved highlighted volcano plot -> {highlight_output}")

    deg_path = args.output.with_name(args.output.stem + "_significant_genes.csv")
    sig = save_significant_genes(df, args.gene_col, args.logfc_col, args.pvalue_col, deg_path)
    print(f"Saved {len(sig)} significant genes -> {deg_path}")


if __name__ == "__main__":
    main()
