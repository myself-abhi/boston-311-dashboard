"""
Build a deploy-ready parquet snapshot of Boston 311 data.

Run this once locally after cloning the repo:

    python build_snapshot.py

It downloads the 5 years of CSVs (if not already cached), trims the columns
the dashboard actually uses, downcasts dtypes (category for strings, int32
for ids), and writes one parquet file per year under data/processed/.

Each per-year parquet is ~10-30 MB after Snappy compression, well under
GitHub's 100 MB-per-file limit. The whole set adds up to ~80-120 MB, which
ships with the repo and lets the deployed Streamlit app load in seconds
with no network calls.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

import data_loader


PROCESSED_DIR = data_loader.DATA_DIR / "processed"

# Columns to keep in the deploy snapshot. Anything not in this list is dropped
# to keep file sizes small.
SNAPSHOT_COLUMNS = [
    "case_enquiry_id",
    "open_dt",
    "closed_dt",
    "department",
    "subject",
    "reason",
    "type",
    "neighborhood",
    "location_zipcode",
    "precinct",
    "latitude",
    "longitude",
    "resolution_hours",
    "log_resolution",
    "slow_case",
    "year",
    "month",
    "week",
    "dayofweek",
]

# Text columns. Kept as plain object on disk - parquet + Snappy already
# dictionary-encodes repeated strings, so we get the size benefit without
# pandas 3.0's strict category-dtype assignment rules biting downstream code.
TEXT_COLUMNS = (
    "department",
    "subject",
    "reason",
    "type",
    "neighborhood",
    "location_zipcode",
    "precinct",
)


def _shrink(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Keep only columns we know about, in the order above.
    keep = [c for c in SNAPSHOT_COLUMNS if c in df.columns]
    df = df[keep]

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("object")

    if "case_enquiry_id" in df.columns:
        # Boston 311 case IDs are 12-digit numbers, well beyond int32 range.
        # Keep them as nullable Int64.
        df["case_enquiry_id"] = pd.to_numeric(df["case_enquiry_id"], errors="coerce").astype("Int64")
    for c in ("year", "month", "week", "dayofweek", "slow_case"):
        if c in df.columns:
            df[c] = df[c].astype("Int16")
    # Keep lat/lon at float64. float32 trims size but Python's stdlib json
    # (used by st.map under the hood) can't serialize numpy.float32 scalars.
    for c in ("latitude", "longitude"):
        if c in df.columns:
            df[c] = df[c].astype("float64")
    for c in ("resolution_hours", "log_resolution"):
        if c in df.columns:
            df[c] = df[c].astype("float32")
    return df


def build() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading full dataset (downloads CSVs if not cached)...")
    df = data_loader.load_data(force_refresh=False, use_cache=True)
    print(f"  rows: {len(df):,}")

    total_bytes = 0
    for year in data_loader.YEARS:
        chunk = df.loc[df["year"] == year].copy()
        chunk = _shrink(chunk)
        dest = PROCESSED_DIR / f"boston_311_{year}.parquet"
        chunk.to_parquet(dest, index=False, compression="snappy")
        size_mb = dest.stat().st_size / (1024 * 1024)
        total_bytes += dest.stat().st_size
        print(f"  {dest.name}: {len(chunk):>7,} rows, {size_mb:5.1f} MB")

    total_mb = total_bytes / (1024 * 1024)
    print(f"\nTotal snapshot size: {total_mb:.1f} MB across {len(data_loader.YEARS)} files")
    print(f"Snapshot directory:  {PROCESSED_DIR}")
    if total_mb > 480:
        print("\nWarning: snapshot is approaching GitHub's 1GB recommended repo size.")
        print("Consider sampling rows or using Git LFS.")


if __name__ == "__main__":
    try:
        build()
    except Exception as exc:
        print(f"Snapshot build failed: {exc}", file=sys.stderr)
        sys.exit(1)
