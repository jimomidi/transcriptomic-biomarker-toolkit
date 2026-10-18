"""
GEO Series Matrix / SOFT / Platform File Preprocessing
========================================================

Utilities for converting raw NCBI GEO download files into tidy, analysis-ready
CSV tables:

  * Series matrix (``*_series_matrix.txt`` / ``.txt.gz``)
      - expression matrix (probes x samples)
      - sample group / phenotype labels (parsed from ``!Sample_characteristics``
        or ``!Sample_group`` rows)
      - technical replicate mapping (detects ``_R1_`` / ``_R2_`` tags in
        supplementary file names)
  * SOFT family file (``*_family.soft.gz``)
      - per-sample metadata table
  * Platform annotation file (e.g. a GPL ``*.txt.gz`` manifest)
      - probe-to-annotation lookup table

All functions operate on file paths and return ``pandas.DataFrame`` objects;
the CLI entry points additionally write the result to CSV.

Expected input
--------------
Raw GEO files as downloaded from https://www.ncbi.nlm.nih.gov/geo/ , placed
under a local directory (see ``--input`` / ``--output`` below). No data is
bundled with this repository.

Usage
-----
    python geo_preprocessing.py series-matrix data/raw/dataset_1_series_matrix.txt.gz --output-dir data/processed
    python geo_preprocessing.py sample-groups data/raw/dataset_1_series_matrix.txt.gz --output-dir data/processed
    python geo_preprocessing.py soft-metadata data/raw/dataset_1_family.soft.gz --output-dir data/processed
    python geo_preprocessing.py platform-annotation data/raw/GPLxxxxx.txt.gz --output-dir data/processed
"""

from __future__ import annotations

import argparse
import gzip
import re
import tempfile
from pathlib import Path
from typing import TextIO

import pandas as pd

REPLICATE_TAG_PATTERN = re.compile(r"_(R\d)_", re.IGNORECASE)


def _open_text(path: Path):
    """Open a plain-text or gzip-compressed text file transparently."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def parse_series_matrix_expression(matrix_path: Path) -> pd.DataFrame:
    """Extract the expression matrix embedded in a GEO series matrix file.

    Reads all lines between ``!series_matrix_table_begin`` and
    ``!series_matrix_table_end`` (or the metadata-comment lines starting
    with ``!``, whichever style the file uses) and parses them as a
    tab-separated table.

    Parameters
    ----------
    matrix_path:
        Path to a ``*_series_matrix.txt`` or ``.txt.gz`` file.

    Returns
    -------
    DataFrame indexed by probe/gene ID with one column per GEO sample (GSM).
    """
    with _open_text(matrix_path) as fh:
        lines = fh.readlines()

    start_idx, end_idx = None, None
    for i, line in enumerate(lines):
        if line.startswith("!series_matrix_table_begin"):
            start_idx = i + 1
        if line.startswith("!series_matrix_table_end"):
            end_idx = i
            break

    if start_idx is not None and end_idx is not None:
        table_lines = lines[start_idx:end_idx]
    else:
        # Fallback: keep every non-metadata line.
        table_lines = [line for line in lines if not line.startswith("!")]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as tmp:
        tmp.writelines(table_lines)
        tmp_path = Path(tmp.name)

    try:
        df_expr = pd.read_csv(tmp_path, sep="\t")
    finally:
        tmp_path.unlink(missing_ok=True)

    if "ID_REF" in df_expr.columns:
        df_expr = df_expr.set_index("ID_REF")
    return df_expr


def detect_technical_replicates(matrix_path: Path) -> pd.DataFrame:
    """Identify technical replicate pairs from supplementary file names.

    GEO submissions sometimes encode technical replicates (e.g. two sequencing
    lanes for the same biological sample) using an ``_R1_`` / ``_R2_`` tag in
    the supplementary file name. This scans ``!Sample_geo_accession`` and
    ``!Sample_supplementary_file`` rows and reports, for every sample, which
    replicate tag (if any) its raw file carries.

    Returns
    -------
    DataFrame with columns ``GSM``, ``file_name``, ``replicate``, ``base_name``.
    """
    with _open_text(matrix_path) as fh:
        lines = fh.readlines()

    gsm_ids: list[str] = []
    file_entries: list[str] = []
    for line in lines:
        if line.startswith("!Sample_geo_accession"):
            parts = line.strip().split("\t")
            gsm_ids = [p.strip().strip('"') for p in parts[1:]]
        elif line.startswith("!Sample_supplementary_file"):
            parts = line.strip().split("\t")
            file_entries.extend(p.strip().strip('"') for p in parts[1:])

    records = []
    for gsm, entry in zip(gsm_ids, file_entries):
        if not entry:
            continue
        for sub_entry in entry.split(";"):
            sub_entry = sub_entry.strip()
            if not sub_entry:
                continue
            file_name = Path(sub_entry).name
            tag_match = REPLICATE_TAG_PATTERN.search(file_name)
            replicate_tag = tag_match.group(1).upper() if tag_match else "NA"
            base_name = REPLICATE_TAG_PATTERN.sub("_", file_name)
            records.append(
                {
                    "GSM": gsm,
                    "file_name": file_name,
                    "replicate": replicate_tag,
                    "base_name": base_name,
                }
            )

    return pd.DataFrame(records)


def extract_sample_groups(matrix_path: Path) -> pd.DataFrame:
    """Extract per-sample group/phenotype labels from a series matrix file.

    Looks for the first ``!Sample_characteristics_ch1`` (or, failing that,
    ``!Sample_group_ch1``) row and returns it alongside the matching
    ``!Sample_geo_accession`` sample IDs.

    Returns
    -------
    DataFrame with columns ``sample_id`` and ``group``.
    """
    with _open_text(matrix_path) as fh:
        lines = fh.readlines()

    sample_ids: list[str] | None = None
    groups: list[str] | None = None

    for line in lines:
        if line.startswith("!Sample_geo_accession"):
            sample_ids = [v.strip().strip('"') for v in line.strip().split("\t")[1:]]
        if groups is None and line.startswith("!Sample_characteristics_ch1"):
            values = line.strip().split("\t")[1:]
            groups = [
                (v.split(":", 1)[-1].strip().strip('"') if ":" in v else v.strip().strip('"'))
                for v in values
            ]
        if groups is None and line.startswith("!Sample_group_ch1"):
            values = line.strip().split("\t")[1:]
            groups = [v.strip().strip('"') for v in values]

    if sample_ids is None or groups is None:
        raise ValueError(
            "Could not locate sample IDs and group labels in the series matrix file. "
            "Inspect the '!Sample_characteristics_ch1' / '!Sample_group_ch1' rows manually."
        )

    return pd.DataFrame({"sample_id": sample_ids, "group": groups})


def parse_soft_family_metadata(soft_path: Path) -> pd.DataFrame:
    """Parse a GEO SOFT family file into a tidy per-sample metadata table.

    Each ``^SAMPLE = GSMxxxxxxx`` block is collected into one row; repeated
    ``!Sample_*`` keys within a block are joined with ``" | "``.
    """
    samples: list[dict] = []
    current: dict | None = None

    def flush_current() -> None:
        if current and "geo_accession" in current:
            for key, value in current.items():
                if isinstance(value, list):
                    current[key] = " | ".join(value)
            samples.append(current.copy())

    with _open_text(soft_path) as fh:
        for line in fh:
            if line.startswith("^SAMPLE ="):
                flush_current()
                current = {}
            elif line.startswith("!Sample_") and current is not None:
                key, _, value = line[1:].partition(" = ")
                key = key.replace("Sample_", "", 1)
                current.setdefault(key, []).append(value.strip().strip('"'))
    flush_current()

    return pd.DataFrame(samples)


def convert_platform_annotation(platform_path: Path) -> pd.DataFrame:
    """Convert a GEO platform (GPL) manifest file into a CSV-friendly table.

    Locates the header row (the first tab-separated line containing a
    ``Probe_Id`` or ``TargetID`` column) and reads everything after it.
    """
    header_idx: int | None = None
    header_cols: list[str] | None = None

    with _open_text(platform_path) as fh:
        for idx, line in enumerate(fh):
            if "\t" in line and re.search(r"\b(Probe_Id|TargetID)\b", line):
                header_idx = idx
                header_cols = line.rstrip("\n").split("\t")
                break

    if header_idx is None or header_cols is None:
        raise ValueError("Could not locate a header row containing 'Probe_Id' or 'TargetID'.")

    return pd.read_csv(
        platform_path,
        sep="\t",
        compression="infer",
        engine="python",
        header=None,
        names=header_cols,
        skiprows=header_idx + 1,
        quoting=3,  # csv.QUOTE_NONE
        on_bad_lines="skip",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("input_file", type=Path, help="Path to the input GEO file.")
        sp.add_argument(
            "--output-dir",
            type=Path,
            default=Path("data/processed"),
            help="Directory to write the resulting CSV into (default: data/processed).",
        )

    sp = sub.add_parser("series-matrix", help="Extract the expression matrix from a series matrix file.")
    add_common(sp)

    sp = sub.add_parser("replicates", help="Detect R1/R2 technical replicates from a series matrix file.")
    add_common(sp)

    sp = sub.add_parser("sample-groups", help="Extract sample group/phenotype labels from a series matrix file.")
    add_common(sp)

    sp = sub.add_parser("soft-metadata", help="Parse a SOFT family file into a per-sample metadata table.")
    add_common(sp)

    sp = sub.add_parser("platform-annotation", help="Convert a GEO platform (GPL) manifest file to CSV.")
    add_common(sp)

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input_file.name.split(".")[0]

    if args.command == "series-matrix":
        df = parse_series_matrix_expression(args.input_file)
        out_path = args.output_dir / f"{stem}_expression_matrix.csv"
        df.to_csv(out_path)
    elif args.command == "replicates":
        df = detect_technical_replicates(args.input_file)
        out_path = args.output_dir / f"{stem}_replicate_mapping.csv"
        df.to_csv(out_path, index=False)
    elif args.command == "sample-groups":
        df = extract_sample_groups(args.input_file)
        out_path = args.output_dir / f"{stem}_sample_groups.csv"
        df.to_csv(out_path, index=False)
    elif args.command == "soft-metadata":
        df = parse_soft_family_metadata(args.input_file)
        out_path = args.output_dir / f"{stem}_metadata.csv"
        df.to_csv(out_path, index=False)
    elif args.command == "platform-annotation":
        df = convert_platform_annotation(args.input_file)
        out_path = args.output_dir / f"{stem}_annotation.csv"
        df.to_csv(out_path, index=False)
    else:  # pragma: no cover - argparse guarantees a valid command
        raise ValueError(f"Unknown command: {args.command}")

    print(f"Saved {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
