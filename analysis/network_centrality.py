"""
Protein-Protein Interaction Network Centrality
========================================================

Builds an undirected graph from a protein-protein (or gene-gene) interaction
edge list and computes standard node centrality metrics: degree, betweenness,
and closeness centrality. These metrics are commonly used to prioritize
"hub" genes within a network of interest (e.g. genes connected to a
differential expression signature).

Expected input
--------------
A CSV with two columns identifying the interacting node pair. By default the
columns are expected to be named ``#node1`` and ``node2`` (as exported by
STRING-DB), but the column names are configurable via ``--node1-col`` /
``--node2-col``.

Usage
-----
    python network_centrality.py data/processed/ppi_network_edges.csv --output results/ppi_centrality.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd


def compute_centrality(edges: pd.DataFrame, node1_col: str, node2_col: str) -> pd.DataFrame:
    """Build a graph from an edge list and compute degree/betweenness/closeness centrality.

    Parameters
    ----------
    edges:
        DataFrame containing at least the two node columns.
    node1_col, node2_col:
        Column names identifying each end of an edge.

    Returns
    -------
    DataFrame with columns ``node``, ``degree``, ``betweenness``, ``closeness``.
    """
    graph = nx.Graph()
    graph.add_edges_from(zip(edges[node1_col], edges[node2_col]))

    degree = dict(graph.degree())
    betweenness = nx.betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)

    return pd.DataFrame(
        {
            "node": list(graph.nodes()),
            "degree": [degree[n] for n in graph.nodes()],
            "betweenness": [betweenness[n] for n in graph.nodes()],
            "closeness": [closeness[n] for n in graph.nodes()],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("edge_list", type=Path, help="CSV file containing the network edge list.")
    parser.add_argument("--node1-col", default="#node1", help="Column name for the first node in each edge.")
    parser.add_argument("--node2-col", default="node2", help="Column name for the second node in each edge.")
    parser.add_argument(
        "--output", type=Path, default=Path("results/network_centrality.csv"), help="Output CSV path."
    )
    args = parser.parse_args()

    edges = pd.read_csv(args.edge_list)
    centrality = compute_centrality(edges, args.node1_col, args.node2_col)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    centrality.to_csv(args.output, index=False)
    print(f"Saved centrality metrics for {len(centrality)} nodes -> {args.output}")


if __name__ == "__main__":
    main()
