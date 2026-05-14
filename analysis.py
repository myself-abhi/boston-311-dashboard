"""
Regression models for Boston 311 case resolution time.

Mirrors the three methods used in the ALY6015 final report:
  1. OLS multiple regression (statsmodels)
  2. LASSO with cross-validated lambda (sklearn LassoCV)
  3. Stepwise forward/backward selection by AIC (manual loop on statsmodels)

All models predict log(1 + resolution_hours) from department, neighborhood,
and optionally year. To keep things tractable on the full 1M-row dataset,
predictors are one-hot encoded with a configurable max number of levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    name: str
    coefficients: pd.DataFrame  # term, estimate, std_error, p_value (where available)
    r2: float
    rmse: float
    n_obs: int
    n_features: int


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

def _top_levels(series: pd.Series, k: int) -> list[str]:
    return series.value_counts(dropna=True).head(k).index.tolist()


def build_design_matrix(
    df: pd.DataFrame,
    include_year: bool = True,
    top_dept: int = 15,
    top_neighborhood: int = 20,
    sample_n: int | None = 200_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, design_meta) ready for the three regressions.

    Categorical levels outside the top-k are collapsed to "Other" so we don't
    blow up the design matrix on noisy long tails.
    """
    work = df.loc[df["resolution_hours"] > 0].copy()

    # pandas 3.0: cast categorical -> object so fillna/assignment with new
    # values ("Unknown", "Other") doesn't raise.
    for col in ("department", "neighborhood"):
        if isinstance(work[col].dtype, pd.CategoricalDtype):
            work[col] = work[col].astype("object")
        else:
            work[col] = work[col].astype("object")
    work["department"] = work["department"].fillna("Unknown")
    work["neighborhood"] = work["neighborhood"].fillna("Unknown")

    top_d = set(_top_levels(work["department"], top_dept))
    top_n = set(_top_levels(work["neighborhood"], top_neighborhood))
    work.loc[~work["department"].isin(top_d), "department"] = "Other"
    work.loc[~work["neighborhood"].isin(top_n), "neighborhood"] = "Other"

    cols = ["department", "neighborhood"]
    if include_year:
        work["year"] = work["year"].astype(int)
        cols.append("year")

    if sample_n is not None and len(work) > sample_n:
        work = work.sample(n=sample_n, random_state=random_state)

    y = np.log1p(work["resolution_hours"].astype(float))

    if include_year:
        dummies = pd.get_dummies(
            work[cols],
            columns=["department", "neighborhood"],
            drop_first=True,
            dtype=float,
        )
        # 'year' stays numeric (continuous trend) to match the report's stepwise model.
        dummies["year"] = work["year"].astype(float).values
    else:
        dummies = pd.get_dummies(
            work[cols],
            columns=["department", "neighborhood"],
            drop_first=True,
            dtype=float,
        )

    meta = pd.DataFrame(
        {
            "feature": dummies.columns,
            "kind": [
                "year" if c == "year"
                else "department" if c.startswith("department_")
                else "neighborhood"
                for c in dummies.columns
            ],
        }
    )

    return dummies.reset_index(drop=True), y.reset_index(drop=True), meta


# -----------------------------------------------------------------------------
# OLS
# -----------------------------------------------------------------------------

def fit_ols(X: pd.DataFrame, y: pd.Series) -> ModelResult:
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y.values, Xc.values).fit()

    coefs = pd.DataFrame(
        {
            "term": Xc.columns,
            "estimate": model.params,
            "std_error": model.bse,
            "p_value": model.pvalues,
        }
    ).reset_index(drop=True)

    y_hat = model.predict(Xc.values)
    rmse = float(np.sqrt(np.mean((y.values - y_hat) ** 2)))

    return ModelResult(
        name="OLS",
        coefficients=coefs,
        r2=float(model.rsquared),
        rmse=rmse,
        n_obs=int(model.nobs),
        n_features=int(X.shape[1]),
    )


# -----------------------------------------------------------------------------
# LASSO
# -----------------------------------------------------------------------------

def fit_lasso(X: pd.DataFrame, y: pd.Series, cv: int = 5, random_state: int = 42) -> ModelResult:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X.values)

    lasso = LassoCV(cv=cv, random_state=random_state, max_iter=5000)
    lasso.fit(X_scaled, y.values)

    # Unscale coefficients so they're interpretable on the original predictors.
    raw_coef = lasso.coef_ / scaler.scale_
    intercept = lasso.intercept_ - np.sum(raw_coef * scaler.mean_)

    coefs = pd.DataFrame(
        {
            "term": ["const"] + list(X.columns),
            "estimate": [intercept] + list(raw_coef),
            "std_error": [np.nan] * (1 + len(X.columns)),
            "p_value": [np.nan] * (1 + len(X.columns)),
        }
    )

    y_hat = lasso.predict(X_scaled)
    rmse = float(np.sqrt(np.mean((y.values - y_hat) ** 2)))
    r2 = float(lasso.score(X_scaled, y.values))

    return ModelResult(
        name="LASSO",
        coefficients=coefs,
        r2=r2,
        rmse=rmse,
        n_obs=int(len(y)),
        n_features=int((np.abs(raw_coef) > 1e-12).sum()),
    )


# -----------------------------------------------------------------------------
# Stepwise (AIC)
# -----------------------------------------------------------------------------

def _aic_of(X_sub: pd.DataFrame, y: pd.Series) -> float:
    Xc = sm.add_constant(X_sub, has_constant="add")
    return float(sm.OLS(y.values, Xc.values).fit().aic)


def fit_stepwise(
    X: pd.DataFrame,
    y: pd.Series,
    max_iter: int = 80,
    verbose: bool = False,
) -> ModelResult:
    """Bidirectional stepwise selection by AIC.

    Walks the feature set adding / dropping the single column that most
    improves AIC. Halts when no move improves the score or after max_iter steps.
    """
    remaining = list(X.columns)
    selected: list[str] = []
    current_aic = _aic_of(X[selected] if selected else pd.DataFrame(index=X.index), y)

    for _ in range(max_iter):
        candidates: list[tuple[float, str, str]] = []  # (aic, action, feature)

        # Try adding each remaining feature.
        for feat in remaining:
            aic = _aic_of(X[selected + [feat]], y)
            candidates.append((aic, "add", feat))

        # Try dropping each currently-selected feature.
        for feat in selected:
            trial = [c for c in selected if c != feat]
            aic = _aic_of(X[trial] if trial else pd.DataFrame(index=X.index), y)
            candidates.append((aic, "drop", feat))

        if not candidates:
            break

        candidates.sort(key=lambda t: t[0])
        best_aic, action, feat = candidates[0]
        if best_aic >= current_aic - 1e-6:
            break

        if action == "add":
            selected.append(feat)
            remaining.remove(feat)
        else:
            selected.remove(feat)
            remaining.append(feat)

        current_aic = best_aic
        if verbose:
            print(f"{action} {feat} -> AIC={current_aic:.2f}")

    Xc = sm.add_constant(X[selected], has_constant="add")
    final = sm.OLS(y.values, Xc.values).fit()

    coefs = pd.DataFrame(
        {
            "term": Xc.columns,
            "estimate": final.params,
            "std_error": final.bse,
            "p_value": final.pvalues,
        }
    ).reset_index(drop=True)

    y_hat = final.predict(Xc.values)
    rmse = float(np.sqrt(np.mean((y.values - y_hat) ** 2)))

    return ModelResult(
        name="Stepwise (AIC)",
        coefficients=coefs,
        r2=float(final.rsquared),
        rmse=rmse,
        n_obs=int(final.nobs),
        n_features=int(len(selected)),
    )


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def run_all_models(
    df: pd.DataFrame,
    include_year: bool = True,
    sample_n: int | None = 50_000,
    stepwise_max_features: int = 15,
    stepwise_max_iter: int = 30,
) -> tuple[dict[str, ModelResult], pd.DataFrame]:
    """Fit OLS, LASSO, and stepwise on the same design matrix.

    Defaults are tuned for Streamlit Cloud's free tier (1 vCPU / 1 GB). The
    full dataset is too costly for stepwise's O(features * iterations) full
    OLS refits before the websocket times out, so we sample and prune.

    Returns (results, comparison_table). The comparison table joins the three
    coefficient tables on `term` so the UI can put them side by side.
    """
    X, y, _ = build_design_matrix(df, include_year=include_year, sample_n=sample_n)

    ols = fit_ols(X, y)
    lasso = fit_lasso(X, y)
    # Stepwise is O(features * iterations) so we cap aggressively.
    if X.shape[1] > stepwise_max_features:
        variances = X.var(axis=0).sort_values(ascending=False)
        keep = variances.head(stepwise_max_features).index.tolist()
        step = fit_stepwise(X[keep], y, max_iter=stepwise_max_iter)
    else:
        step = fit_stepwise(X, y, max_iter=stepwise_max_iter)

    comparison = (
        ols.coefficients[["term", "estimate"]]
        .rename(columns={"estimate": "OLS"})
        .merge(
            lasso.coefficients[["term", "estimate"]].rename(columns={"estimate": "LASSO"}),
            on="term",
            how="outer",
        )
        .merge(
            step.coefficients[["term", "estimate"]].rename(columns={"estimate": "Stepwise"}),
            on="term",
            how="outer",
        )
        .fillna(0.0)
    )

    return {"OLS": ols, "LASSO": lasso, "Stepwise": step}, comparison
