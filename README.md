# Transcriptomic Biomarker Toolkit

A collection of Python scripts for identifying and characterizing candidate
gene expression biomarkers from public case-control microarray datasets
(e.g. from NCBI GEO). It covers the full path from raw platform files to
publication-style figures: preprocessing, differential expression
summarization, single-gene classification performance, network centrality,
cross-condition overlap analysis, and enrichment/UpSet visualization.

The toolkit is dataset-agnostic: every script takes file paths and column
names as arguments (or as an editable configuration block), so it can be
pointed at any similarly-structured case-control expression dataset. No data
is bundled with this repository.

## Repository structure

```
transcriptomic-biomarker-toolkit/
├── analysis/
│   ├── geo_preprocessing.py          # GEO series matrix / SOFT / platform file -> tidy CSV
│   ├── biomarker_classification.py   # per-gene XGBoost classification (SMOTE + ROC-AUC)
│   ├── condition_overlap_analysis.py # shared drugs/indications across conditions
│   └── network_centrality.py         # PPI network degree/betweenness/closeness
├── visualization/
│   ├── volcano_plot.py               # DE volcano plot (standard + gene-highlighted)
│   ├── correlation_heatmap.py        # case vs. control gene correlation heatmaps
│   ├── pathway_enrichment_bubble_plot.py  # KEGG/Reactome/GO bubble plot
│   └── upset_drug_interaction_plot.py     # UpSet plot of shared drug interactions
├── requirements.txt
├── LICENSE
└── .gitignore
```

Scripts under `analysis/` compute results (tables, statistics, model
metrics); scripts under `visualization/` turn already-computed data into
figures. `biomarker_classification.py` and `network_centrality.py` also
save a plot alongside their primary CSV output, since a plot is the natural
summary of those results.

## Workflow

A typical end-to-end run looks like this:

1. **Preprocess raw GEO files** (`analysis/geo_preprocessing.py`) into an
   expression matrix, a sample group/annotation table, and (optionally) a
   platform annotation or technical-replicate table.
2. **Summarize differential expression** (`visualization/volcano_plot.py`)
   from a precomputed per-gene statistics table (see below), producing a
   volcano plot and a CSV of significant genes.
3. **Evaluate single-gene classification performance**
   (`analysis/biomarker_classification.py`) for a candidate biomarker panel
   across one or more datasets, producing ROC curves, AUC, and a t-test
   p-value per gene.
4. **Inspect co-expression structure** (`visualization/correlation_heatmap.py`)
   of the candidate panel, separately in cases and controls.
5. **Characterize network context** (`analysis/network_centrality.py`) of
   candidate genes within a protein-protein interaction network.
6. **Compare across conditions**: find drugs/indications shared between
   conditions (`analysis/condition_overlap_analysis.py`), visualize shared
   drug-interaction scores as an UpSet plot
   (`visualization/upset_drug_interaction_plot.py`), and visualize shared
   pathway enrichment as a bubble plot
   (`visualization/pathway_enrichment_bubble_plot.py`).

Each step is independent and can be run on its own -- the scripts are not
wired into a single pipeline driver, since in practice the inputs to steps
2-6 typically come from a mix of this toolkit's own outputs and external
tools (e.g. limma/GEO2R for differential expression statistics, or DAVID
for pathway enrichment).

## Requirements

Python 3.10+ and the packages listed in `requirements.txt`:

```
pandas>=2.0,<3.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
upsetplot>=0.9
scikit-learn>=1.3
xgboost>=2.0
imbalanced-learn>=0.11
networkx>=3.0
```

### Installation

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Expected input formats

No datasets are included in this repository. Place your own input files
under a local `data/raw/` directory (or any path of your choosing) and pass
their paths on the command line.

| Data type | Format | Used by |
|---|---|---|
| GEO series matrix | `*_series_matrix.txt` or `.txt.gz`, as downloaded from GEO | `geo_preprocessing.py` |
| GEO SOFT family file | `*_family.soft.gz` | `geo_preprocessing.py` |
| GEO platform (GPL) file | tab-separated manifest with a `Probe_Id`/`TargetID` header row | `geo_preprocessing.py` |
| Expression matrix | CSV, probes/genes (rows) x samples (columns), first column is the feature ID | `biomarker_classification.py`, `correlation_heatmap.py` |
| Sample annotation | CSV with a sample ID column and a group/label column | `biomarker_classification.py`, `correlation_heatmap.py` |
| Differential expression results | CSV with a gene ID column, `logFC`, and a p-value column (e.g. `P.Value` or `adj.P.Val`) -- produced upstream by a tool such as limma or GEO2R; this toolkit does not compute the statistics itself | `volcano_plot.py` |
| Network edge list | CSV with two node-ID columns (default names `#node1`, `node2`, as exported by STRING-DB) | `network_centrality.py` |
| Drug interaction table | CSV with `drug`, `indication`, and (for the UpSet plot) `interaction score` columns | `condition_overlap_analysis.py`, `upset_drug_interaction_plot.py` |
| Enrichment results | CSV with `Term`, `Fold Enrichment`, `Count`, `Note` columns, as exported by an enrichment tool such as DAVID | `pathway_enrichment_bubble_plot.py` |

## Usage

All scripts print `--help` for the full list of options. Representative
examples:

```bash
# GEO preprocessing
python analysis/geo_preprocessing.py series-matrix data/raw/dataset_1_series_matrix.txt.gz --output-dir data/processed
python analysis/geo_preprocessing.py sample-groups data/raw/dataset_1_series_matrix.txt.gz --output-dir data/processed
python analysis/geo_preprocessing.py replicates data/raw/dataset_1_series_matrix.txt.gz --output-dir data/processed
python analysis/geo_preprocessing.py soft-metadata data/raw/dataset_1_family.soft.gz --output-dir data/processed
python analysis/geo_preprocessing.py platform-annotation data/raw/GPLxxxxx.txt.gz --output-dir data/processed

# Volcano plot + significant gene table
python visualization/volcano_plot.py data/processed/dataset_1_de_results.csv \
    --gene-col gene --pvalue-col P.Value \
    --logfc-threshold 0.3 --pvalue-threshold 0.05 \
    --output results/dataset_1_volcano.png

# ...optionally, also render a panel-highlighted volcano plot in the same run
python visualization/volcano_plot.py data/processed/dataset_1_de_results.csv \
    --highlight-genes GENE_A=probe_1 GENE_B=probe_2 \
    --output results/dataset_1_volcano.png

# Single-gene biomarker classification (edit DATASET_CONFIG in the script first)
python analysis/biomarker_classification.py --output-dir results/

# Case vs. control correlation heatmaps for a gene panel
python visualization/correlation_heatmap.py \
    data/processed/dataset_1_expression_matrix.csv data/processed/dataset_1_annotation.csv \
    --probe-map GENE_A=probe_1 GENE_B=probe_2 \
    --output-dir results/

# Network centrality
python analysis/network_centrality.py data/processed/ppi_network_edges.csv --output results/ppi_centrality.csv

# Cross-condition overlap
python analysis/condition_overlap_analysis.py \
    --input condition_1=data/processed/condition_1_interactions.csv \
    --input condition_2=data/processed/condition_2_interactions.csv \
    --input condition_3=data/processed/condition_3_interactions.csv \
    --column indication --output-dir results/

# UpSet plot of shared drug-interaction scores
python visualization/upset_drug_interaction_plot.py \
    --input "condition_1 & condition_2"=data/processed/common_1_2.csv \
    --input "condition_2 & condition_3"=data/processed/common_2_3.csv \
    --input "condition_1 & condition_3"=data/processed/common_1_3.csv \
    --input "condition_1 & condition_2 & condition_3"=data/processed/common_1_2_3.csv \
    --output results/upset_drug_interactions.png

# Pathway enrichment bubble plot
python visualization/pathway_enrichment_bubble_plot.py data/processed/kegg_enrichment.csv \
    --title "KEGG Pathways" --term-separator ":" --output results/kegg_pathways.png
```

## Outputs

Every script writes CSV tables and/or PNG figures to the directory given by
`--output` / `--output-dir` (default: `results/`), which is created
automatically if it does not exist. Nothing is written outside the path(s)
you specify.

## Reproducibility

* All stochastic steps (SMOTE resampling, train/test splitting, the XGBoost
  classifier) use a fixed `random_state=42`.
* All file paths are relative arguments/configuration -- no machine-specific
  absolute paths are hard-coded anywhere in the codebase.
* Dependency versions are pinned to tested ranges in `requirements.txt`.

## Customization

* **Biomarker panel / datasets**: edit `DATASET_CONFIG` at the top of
  `analysis/biomarker_classification.py` (or import `DatasetSpec` and
  `run_analysis` and build your own list of datasets programmatically).
* **Significance thresholds**: `--logfc-threshold` / `--pvalue-threshold`
  on `volcano_plot.py` (defaults: `|logFC| > 0.3`, `p < 0.05`).
* **Gene panel for correlation heatmaps**: `--probe-map` on
  `correlation_heatmap.py`.
* **Number of conditions**: `condition_overlap_analysis.py` and
  `upset_drug_interaction_plot.py` both accept any number of `--input`
  entries (2 or more).

## Known issues

* `upsetplot` 0.9.x renders incorrectly under `pandas` 3.x (an internal
  chained-assignment `fillna` call becomes a no-op under pandas'
  Copy-on-Write behavior, which can raise a `ValueError: Invalid RGBA
  argument: nan` when saving the figure). `requirements.txt` pins
  `pandas<3.0` to avoid this; if you upgrade `pandas` independently, check
  for an `upsetplot` release that addresses this before upgrading it too.

## License

Released under the MIT License -- see [LICENSE](LICENSE).
