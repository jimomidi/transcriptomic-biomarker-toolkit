"""
Single-Gene Biomarker Classification (XGBoost + SMOTE)
========================================================

Evaluates how well the expression level of each candidate gene, taken alone,
discriminates case samples from control samples in one or more case-control
expression datasets.

For every (dataset, gene) pair:

  1. The corresponding probe's expression values are extracted and
     z-score standardized.
  2. Class imbalance is corrected with SMOTE oversampling.
  3. The resampled data is split into train/test sets (80/20, stratified).
  4. An XGBoost classifier is trained and evaluated by ROC-AUC on the held-out
     test set.
  5. A two-sample Welch's t-test on the *raw* (pre-resampling) expression
     values provides a p-value for the case/control difference.

Results are summarized as one ROC panel figure (one subplot per gene per
dataset) and, optionally, a CSV table of AUC / p-value per gene.

Expected input
--------------
For each dataset:
  * An expression matrix CSV, probes (rows) x samples (columns), with the
    probe/feature ID as the index column.
  * An annotation CSV with a sample identifier column and a group/label
    column indicating case vs. control status.

Configure the datasets to analyze, and the gene -> probe ID map for each, in
``DATASET_CONFIG`` below (or supply your own dict programmatically).

Usage
-----
    python biomarker_classification.py --output-dir results/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_ind
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass
class DatasetSpec:
    """Configuration for one case-control expression dataset."""

    name: str
    expression_matrix: Path
    annotation: Path
    genes: dict[str, str]  # gene symbol -> probe/feature ID
    case_label: str
    control_keywords: Sequence[str] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Example configuration. Replace with your own datasets and gene panels, or
# construct a list[DatasetSpec] programmatically and pass it to run_analysis().
# ---------------------------------------------------------------------------
DATASET_CONFIG: list[DatasetSpec] = [
    DatasetSpec(
        name="dataset_1",
        expression_matrix=Path("data/processed/dataset_1_expression_matrix.csv"),
        annotation=Path("data/processed/dataset_1_annotation.csv"),
        genes={
            "GENE_A": "probe_id_1",
            "GENE_B": "probe_id_2",
            "GENE_C": "probe_id_3",
        },
        case_label="case",
        control_keywords=["control"],
    ),
]


def _load_labels(annotation_path: Path, case_label: str, control_keywords: Sequence[str]) -> pd.Series:
    """Load an annotation CSV and derive a binary case(1)/control(0) label per sample.

    Accepts a few common annotation layouts:
      * columns ``sample_id`` + ``annotation`` (annotation text containing
        the case label, e.g. free-text phenotype descriptions)
      * columns ``sample_id`` + ``group``
      * columns ``sample_id`` + ``<some other name>`` (first non-ID column is
        used)
    """
    ann = pd.read_csv(annotation_path)
    ann.columns = [str(c).strip() for c in ann.columns]

    id_col = "sample_id" if "sample_id" in ann.columns else ann.columns[0]
    if "group" in ann.columns:
        value_col = "group"
    elif "annotation" in ann.columns:
        value_col = "annotation"
    else:
        value_col = ann.columns[1]

    ann = ann[[id_col, value_col]].rename(columns={id_col: "sample_id", value_col: "group"})
    ann["sample_id"] = ann["sample_id"].astype(str).str.replace('"', "", regex=False)

    def classify(value: str) -> float:
        text = str(value).strip().lower()
        if text == case_label.lower() or case_label.lower() in text:
            return 1.0
        if any(kw.lower() in text for kw in control_keywords):
            return 0.0
        return np.nan

    ann["label"] = ann["group"].apply(classify)
    ann = ann.dropna(subset=["label"])
    ann["label"] = ann["label"].astype(int)
    return ann.set_index("sample_id")["label"]


def evaluate_gene(expr_row: np.ndarray, labels: np.ndarray) -> dict | None:
    """Fit an XGBoost classifier on a single gene's expression and score it.

    Returns a dict with ROC curve data, AUC, and a t-test p-value, or ``None``
    if the sample has fewer than two label classes.
    """
    if len(np.unique(labels)) < 2:
        return None

    X = expr_row.reshape(-1, 1)
    y = labels

    group_0 = X[y == 0].flatten()
    group_1 = X[y == 1].flatten()
    _, pval = ttest_ind(group_0, group_1, equal_var=False)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=TEST_SIZE, stratify=y_resampled, random_state=RANDOM_STATE
    )

    model = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    return {"fpr": fpr, "tpr": tpr, "auc": roc_auc, "pval": pval}


def run_analysis(datasets: list[DatasetSpec]) -> list[dict]:
    """Run single-gene classification for every gene in every dataset.

    Returns a list of result dicts (one per successfully evaluated gene),
    each augmented with ``gene`` and ``dataset`` keys.
    """
    all_results: list[dict] = []

    for spec in datasets:
        expr = pd.read_csv(spec.expression_matrix, index_col=0)
        expr.index = expr.index.astype(str).str.replace('"', "", regex=False)
        expr.columns = expr.columns.astype(str).str.replace('"', "", regex=False)

        labels = _load_labels(spec.annotation, spec.case_label, spec.control_keywords)

        common_samples = expr.columns.intersection(labels.index)
        expr_matched = expr[common_samples]
        labels_matched = labels[common_samples]

        for gene_name, probe_id in spec.genes.items():
            if probe_id not in expr_matched.index:
                print(f"Warning: probe '{probe_id}' for gene '{gene_name}' not found in {spec.name}.")
                continue

            result = evaluate_gene(expr_matched.loc[probe_id].values, labels_matched.values)
            if result is None:
                print(f"Warning: only one class present for gene '{gene_name}' in {spec.name}; skipped.")
                continue

            result.update(gene=gene_name, dataset=spec.name)
            all_results.append(result)

    return all_results


def plot_roc_panel(results: list[dict], output_path: Path, n_cols: int = 4) -> None:
    """Render a grid of ROC curves, one subplot per (dataset, gene) result."""
    if not results:
        print("No results to plot.")
        return

    n_rows = int(np.ceil(len(results) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, result in enumerate(results):
        ax = axes[i]
        ax.plot(result["fpr"], result["tpr"], color="purple", lw=2,
                label=f"AUC = {result['auc']:.2f}\nP = {result['pval']:.2e}")
        ax.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{result['gene']} ({result['dataset']})")
        ax.legend(loc="lower right", fontsize=9, frameon=False)

    for j in range(len(results), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC panel -> {output_path}")


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"dataset": r["dataset"], "gene": r["gene"], "auc": r["auc"], "pval": r["pval"]} for r in results]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"), help="Directory to write outputs into (default: results)."
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = run_analysis(DATASET_CONFIG)

    summary = results_to_dataframe(results)
    summary_path = args.output_dir / "biomarker_classification_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary table -> {summary_path}")

    plot_roc_panel(results, args.output_dir / "roc_panel.png")


if __name__ == "__main__":
    main()
