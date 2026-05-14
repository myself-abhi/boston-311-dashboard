"""
Pre-compute the OLS, LASSO, and Stepwise model results offline.

Run locally once with full compute budget:

    python precompute_models.py

The result is written to data/processed/model_results.json (~50 KB). The
deployed Streamlit app loads this file and displays it instantly, so the
free-tier server never has to run a heavy fit.

To refresh after a snapshot change:
    python precompute_models.py
    git add data/processed/model_results.json
    git commit -m "Refresh model results"
    git push
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import analysis
import data_loader


OUTPUT_PATH = data_loader.DATA_DIR / "processed" / "model_results.json"

# Full compute budget - this runs locally so we don't have to compromise.
SAMPLE_N = 200_000
STEPWISE_MAX_FEATURES = 25
STEPWISE_MAX_ITER = 60


def _model_to_dict(result) -> dict:
    return {
        "name": result.name,
        "r2": float(result.r2),
        "rmse": float(result.rmse),
        "n_obs": int(result.n_obs),
        "n_features": int(result.n_features),
        "coefficients": result.coefficients.to_dict(orient="records"),
    }


def build() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading full dataset...")
    df = data_loader.load_data()
    print(f"  rows: {len(df):,}")

    print(f"\nFitting models on {SAMPLE_N:,}-row sample with full compute budget...")
    results, comparison = analysis.run_all_models(
        df,
        include_year=True,
        sample_n=SAMPLE_N,
        stepwise_max_features=STEPWISE_MAX_FEATURES,
        stepwise_max_iter=STEPWISE_MAX_ITER,
    )

    for name, r in results.items():
        print(f"  {name:>10s}  R2={r.r2:.4f}  RMSE={r.rmse:.3f}  features={r.n_features}")

    payload = {
        "metadata": {
            "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sample_n": SAMPLE_N,
            "stepwise_max_features": STEPWISE_MAX_FEATURES,
            "stepwise_max_iter": STEPWISE_MAX_ITER,
            "n_rows_used": int(len(df)),
            "year_range": [
                int(df["year"].min()),
                int(df["year"].max()),
            ],
            "include_year": True,
        },
        "results": {name: _model_to_dict(r) for name, r in results.items()},
        "comparison": comparison.to_dict(orient="records"),
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nWrote {OUTPUT_PATH} ({size_kb:.1f} KB)")
    print(
        "\nNext steps:\n"
        f"  git add {OUTPUT_PATH.relative_to(data_loader.PROJECT_ROOT)}\n"
        "  git commit -m 'Refresh pre-computed model results'\n"
        "  git push"
    )


if __name__ == "__main__":
    build()
