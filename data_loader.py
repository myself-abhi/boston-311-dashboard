"""
Boston 311 Service Requests data loader.

Downloads yearly CSVs from data.boston.gov (CKAN API) on first run,
caches them on disk, and returns a single combined DataFrame.

Behaviour:
  1. On first run, fetches each year's CSV (2015-2019) and writes to data/raw/.
  2. Cleans and unions the years (the equivalent of dplyr::bind_rows).
  3. Writes a parquet cache at data/boston_311_2015_2019.parquet for fast reloads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

YEARS: tuple[int, ...] = (2015, 2016, 2017, 2018, 2019)

# Verified URLs from data.boston.gov CKAN API (May 2026 snapshot).
# The loader will try CKAN first, and fall back to these if CKAN fails.
FALLBACK_URLS: dict[int, str] = {
    2015: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/c9509ab4-6f6d-4b97-979a-0cf2a10c922b/download/tmphrybkxuh.csv",
    2016: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/b7ea6b1b-3ca4-4c5b-9713-6dc1db52379a/download/tmpzxzxeqfb.csv",
    2017: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/30022137-709d-465e-baae-ca155b51927d/download/tmpzccn8u4q.csv",
    2018: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/2be28d90-3a90-4af1-a3f6-f28c1e25880a/download/tmp7602cia8.csv",
    2019: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/ea2e4696-4a2d-429c-9807-d02eb92e0222/download/tmpcje3ep_w.csv",
}

CKAN_PACKAGE_URL = "https://data.boston.gov/api/3/action/package_show?id=311-service-requests"

# Columns we keep from the raw CSVs. Boston 311 CSVs include far more columns,
# but these are the ones used in the analysis.
KEEP_COLUMNS: list[str] = [
    "case_enquiry_id",
    "open_dt",
    "closed_dt",
    "case_status",
    "department",
    "subject",
    "reason",
    "type",
    "neighborhood",
    "location_zipcode",
    "precinct",
    "latitude",
    "longitude",
    "source",
]


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_PATH = DATA_DIR / "boston_311_2015_2019.parquet"


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Download helpers
# -----------------------------------------------------------------------------

def _resolve_urls_from_ckan() -> dict[int, str]:
    """Query the CKAN API to find the current resource URL for each year."""
    try:
        resp = requests.get(CKAN_PACKAGE_URL, timeout=20)
        resp.raise_for_status()
        resources = resp.json()["result"]["resources"]
    except Exception:
        return {}

    found: dict[int, str] = {}
    for r in resources:
        if (r.get("format") or "").upper() != "CSV":
            continue
        name = (r.get("name") or "").lower()
        for year in YEARS:
            if str(year) in name and year not in found:
                found[year] = r["url"]
                break
    return found


def _download_year(year: int, url: str, dest: Path) -> None:
    """Stream a CSV to disk so memory stays low even for ~200MB files."""
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)


def download_year_csvs(years: Iterable[int] = YEARS, force: bool = False) -> dict[int, Path]:
    """Ensure each year's raw CSV is on disk; return mapping year -> path."""
    _ensure_dirs()

    ckan_urls = _resolve_urls_from_ckan()
    resolved: dict[int, str] = {}
    for year in years:
        resolved[year] = ckan_urls.get(year, FALLBACK_URLS[year])

    paths: dict[int, Path] = {}
    for year, url in resolved.items():
        dest = RAW_DIR / f"311_{year}.csv"
        if force or not dest.exists() or dest.stat().st_size < 1024:
            _download_year(year, url, dest)
        paths[year] = dest
    return paths


# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------

def _read_one_year(path: Path) -> pd.DataFrame:
    """Read a single year's CSV, keep relevant columns, parse timestamps."""
    # Peek at the header to figure out which of KEEP_COLUMNS actually exist
    sample = pd.read_csv(path, nrows=1)
    available = [c for c in KEEP_COLUMNS if c in sample.columns]

    df = pd.read_csv(
        path,
        usecols=available,
        dtype=str,
        low_memory=False,
    )
    for col in ("open_dt", "closed_dt"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "latitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    if "longitude" in df.columns:
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df


def _clean_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Compute resolution_hours, filter to closed cases, derive year/month."""
    df = df.copy()

    # Resolution time in hours (continuous outcome from the report).
    df["resolution_hours"] = (df["closed_dt"] - df["open_dt"]).dt.total_seconds() / 3600.0

    # Keep only closed cases with non-negative resolution times.
    df = df.loc[df["resolution_hours"].notna() & (df["resolution_hours"] >= 0)]

    df["year"] = df["open_dt"].dt.year.astype("Int64")
    df["month"] = df["open_dt"].dt.month.astype("Int64")
    df["week"] = df["open_dt"].dt.isocalendar().week.astype("Int64")
    df["dayofweek"] = df["open_dt"].dt.dayofweek.astype("Int64")

    # Tidy text columns (strip + replace empties with NaN).
    for col in ("department", "subject", "reason", "type", "neighborhood",
                "location_zipcode", "precinct", "case_status", "source"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().replace({"": pd.NA})

    # Binary 'slow case' flag: above-mean resolution time (matches the report).
    overall_mean = df["resolution_hours"].mean()
    df["slow_case"] = (df["resolution_hours"] > overall_mean).astype(int)

    # Log outcome (the regressions in the report are on log resolution hours).
    df["log_resolution"] = np.log1p(df["resolution_hours"])

    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def _processed_snapshot_paths() -> list[Path]:
    if not PROCESSED_DIR.exists():
        return []
    return sorted(PROCESSED_DIR.glob("boston_311_*.parquet"))


def _load_from_processed(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    # Ensure datetime types survive the round trip (parquet preserves them, but
    # be defensive in case the snapshot was produced by a different writer).
    for col in ("open_dt", "closed_dt"):
        if col in combined.columns and not pd.api.types.is_datetime64_any_dtype(combined[col]):
            combined[col] = pd.to_datetime(combined[col], errors="coerce")
    return combined


def load_data(force_refresh: bool = False, use_cache: bool = True) -> pd.DataFrame:
    """Return the combined, cleaned 2015-2019 Boston 311 DataFrame.

    Resolution order:
      1. data/processed/boston_311_<year>.parquet snapshot (shipped with repo)
      2. data/boston_311_2015_2019.parquet local cache (built on first run)
      3. Live download from data.boston.gov, then build the local cache.
    """
    _ensure_dirs()

    # 1. Deploy-friendly snapshot if it was committed.
    processed_paths = _processed_snapshot_paths()
    if processed_paths and not force_refresh:
        return _load_from_processed(processed_paths)

    # 2. Local parquet cache from a previous full load.
    if use_cache and CACHE_PATH.exists() and not force_refresh:
        return pd.read_parquet(CACHE_PATH)

    # 3. Live download fallback.
    csv_paths = download_year_csvs(force=force_refresh)
    frames: list[pd.DataFrame] = []
    for year in YEARS:
        frame = _read_one_year(csv_paths[year])
        # Defensive: enforce year tag in case open_dt parsing was lossy.
        frame["_source_year"] = year
        frames.append(frame)

    # bind_rows() equivalent.
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = _clean_combined(combined)

    try:
        combined.to_parquet(CACHE_PATH, index=False)
    except Exception:
        # Parquet write is best-effort; the app still works without the cache.
        pass

    return combined


def dataset_summary(df: pd.DataFrame) -> dict[str, float | int]:
    """Quick descriptive stats used by the dashboard KPI cards."""
    res = df["resolution_hours"]
    return {
        "total_cases": int(len(df)),
        "mean_hours": float(res.mean()),
        "median_hours": float(res.median()),
        "sd_hours": float(res.std()),
        "min_hours": float(res.min()),
        "max_hours": float(res.max()),
        "slow_share": float(df["slow_case"].mean()),
    }


if __name__ == "__main__":
    # Allows: python data_loader.py to prefetch and build the cache.
    df = load_data()
    print(df.shape)
    print(df.head(3))
    print(dataset_summary(df))
