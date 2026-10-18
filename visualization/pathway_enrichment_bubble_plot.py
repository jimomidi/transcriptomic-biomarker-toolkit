"""
Pathway Enrichment Bubble ("Lollipop") Plot
========================================================

Renders a horizontal bubble/lollipop chart of functional enrichment results
(e.g. KEGG pathways, Reactome pathways, or GO Biological Process terms), as
commonly exported by enrichment tools such as DAVID. Each row is one
enriched term; bubble position encodes fold enrichment, bubble size encodes
gene count, and bubble/background color encodes a grouping column (e.g.
which comparison group the shared pathway came from).

Expected input
--------------
A CSV with at least the columns:
  * ``Term``              -- pathway/term identifier and description
                              (e.g. "hsa04659:Th17 cell differentiation" for
                              KEGG, or "R-HSA-1234~Some Reactome pathway")
  * ``Fold Enrichment``    -- numeric enrichment score
  * ``Count``              -- number of genes contributing to the term
  * ``Note``                -- group label used for color-coding and the
                              legend (e.g. which condition comparison the
                              row belongs to); forward-filled if blank

Usage
-----
    python pathway_enrichment_bubble_plot.py data/processed/kegg_enrichment.csv \\
        --title "KEGG Pathways" --term-separator ":" --output results/kegg_pathways.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_enrichment_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Note"] = df["Note"].ffill()
    df["Fold Enrichment"] = pd.to_numeric(df["Fold Enrichment"], errors="coerce")
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce")
    df = df[df["Fold Enrichment"].notnull() & df["Count"].notnull()].copy()
    df["Note"] = df["Note"].str.replace("Shared pathways between ", "", regex=False)
    return df


def plot_enrichment_bubbles(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    term_separator: str = ":",
    figsize: tuple[float, float] = (11.5, 7),
    dpi: int = 300,
) -> None:
    """Draw the bubble plot and save it to ``output_path``.

    ``term_separator`` controls how the human-readable label is extracted
    from the ``Term`` column: everything after the first occurrence of the
    separator is used (e.g. ``":"`` for "hsa04659:Th17 cell differentiation",
    ``"~"`` for "R-HSA-1234~Some Reactome pathway").
    """
    df = df.copy()
    df["Y_Label"] = df["Term"].str.split(term_separator, n=1).str[1]
    df["Y_Pos"] = range(len(df))

    unique_notes = df["Note"].unique()
    palette = sns.color_palette("pastel", len(unique_notes))
    bg_color_map = dict(zip(unique_notes, palette))

    sns.set(style="white")
    fig, ax = plt.subplots(figsize=figsize)

    for _, row in df.iterrows():
        bg_color = bg_color_map.get(row["Note"], "#f5f5f5")
        ax.add_patch(patches.Rectangle(
            (0, row["Y_Pos"] - 0.5), df["Fold Enrichment"].max() + 1, 1,
            color=bg_color, alpha=0.3, zorder=0,
        ))

    for _, row in df.iterrows():
        ax.hlines(
            y=row["Y_Pos"], xmin=0, xmax=row["Fold Enrichment"] - 0.1,
            color="black", linestyle=(0, (6, 4)), linewidth=2.0, zorder=1,
        )

    sns.scatterplot(
        data=df, x="Fold Enrichment", y="Y_Pos", size="Count", hue="Note",
        palette=bg_color_map, sizes=(100, 1200), edgecolor="black",
        alpha=1.0, legend="full", zorder=2, ax=ax,
    )

    legend = ax.legend(title=" ", bbox_to_anchor=(1.03, 1), loc="upper left", borderaxespad=0, frameon=False)
    legend.set_title(" ")

    ax.set_yticks(df["Y_Pos"])
    ax.set_yticklabels(df["Y_Label"], fontsize=13)
    ax.set_xlabel("Fold Enrichment", fontsize=14, labelpad=15, fontweight="bold")
    ax.set_ylabel("")
    ax.set_title("")
    ax.text(0.5, 1.0, title, fontsize=18, fontweight="bold", ha="center", va="bottom", transform=ax.transAxes)

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", length=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("enrichment_table", type=Path, help="CSV of enrichment results (see module docstring).")
    parser.add_argument("--title", required=True, help='Plot title, e.g. "KEGG Pathways".')
    parser.add_argument(
        "--term-separator", default=":",
        help="Separator used to split the pathway ID from its description in the 'Term' column (default: ':').",
    )
    parser.add_argument("--output", type=Path, default=Path("results/pathway_enrichment.png"))
    args = parser.parse_args()

    df = load_enrichment_table(args.enrichment_table)
    plot_enrichment_bubbles(df, args.output, args.title, term_separator=args.term_separator)
    print(f"Saved pathway enrichment plot -> {args.output}")


if __name__ == "__main__":
    main()
