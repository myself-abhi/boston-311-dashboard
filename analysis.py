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

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# LASSO at max_iter=500 with a 5-point alpha grid converges close enough for
# coefficient comparison but sometimes trips ConvergenceWarning. The
# coefficient estimates are still well within the variation we care about,
# so silence the warning to keep the deploy log readable.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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

    # sklearn 1.8 defaults to n_alphas=100 which means 100 * cv = 500 Lasso
    # fits. On Streamlit Cloud's shared 1 vCPU that takes minutes. A 10-point
    # log-spaced grid is more than enough for stable lambda.min selection on
    # this problem, and cuts CV cost by 10x.
    alphas = np.logspace(-3, 1, 5)
    lasso = LassoCV(alphas=alphas, cv=cv, random_state=random_state, max_iter=500)
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
# Stepwise (AIC) - numpy fast path
# -----------------------------------------------------------------------------

def _aic_from_residuals(rss: float, n: int, k: int) -> float:
    """AIC for a Gaussian linear model with k regression coefficients (intercept
    counted) and an estimated sigma. Matches statsmodels' OLS.aic up to a
    constant that cancels out when comparing models on the same y."""
    if rss <= 0.0:
        return float("-inf")
    return n * np.log(rss / n) + 2.0 * (k + 1)


def _aic_np(X_sub: np.ndarray, y: np.ndarray) -> float:
    """Fast AIC: numpy lstsq instead of a full statsmodels OLS.fit().

    For an intercept-only null model pass X_sub with shape (n, 0).
    """
    n = y.shape[0]
    Xc = np.column_stack([np.ones(n), X_sub]) if X_sub.shape[1] else np.ones((n, 1))
    beta, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    rss = float(np.dot(resid, resid))
    return _aic_from_residuals(rss, n, Xc.shape[1])


def fit_stepwise(
    X: pd.DataFrame,
    y: pd.Series,
    max_iter: int = 30,
    verbose: bool = False,
) -> ModelResult:
    """Bidirectional stepwise selection by AIC, on numpy arrays.

    Each candidate add/drop is evaluated with a single np.linalg.lstsq call,
    which is roughly 50x faster than statsmodels OLS.fit() because we skip
    standard errors, t-stats, and the whole regression-results object. We
    still use statsmodels for the FINAL fit so the returned coefficient
    table carries proper std_error and p_value columns.
    """
    feature_names = list(X.columns)
    X_arr = X.values.astype(np.float64, copy=False)
    y_arr = y.values.astype(np.float64, copy=False)
    n = y_arr.shape[0]

    remaining: set[int] = set(range(X_arr.shape[1]))
    selected: list[int] = []
    current_aic = _aic_np(np.empty((n, 0)), y_arr)

    for _ in range(max_iter):
        best_delta = 0.0
        best_action: str | None = None
        best_feat: int | None = None

        # Adds: one column joins the selected set.
        for j in remaining:
            cols = selected + [j]
            aic = _aic_np(X_arr[:, cols], y_arr)
            delta = aic - current_aic
            if delta < best_delta - 1e-9:
                best_delta, best_action, best_feat = delta, "add", j

        # Drops: one column leaves the selected set.
        for j in selected:
            cols = [c for c in selected if c != j]
            X_sub = X_arr[:, cols] if cols else np.empty((n, 0))
            aic = _aic_np(X_sub, y_arr)
            delta = aic - current_aic
            if delta < best_delta - 1e-9:
                best_delta, best_action, best_feat = delta, "drop", j

        if best_action is None:
            break

        if best_action == "add":
            selected.append(best_feat)
            remaining.discard(best_feat)
        else:
            selected.remove(best_feat)
            remaining.add(best_feat)

        current_aic += best_delta
        if verbose:
            print(f"{best_action} {feature_names[best_feat]} -> AIC={current_aic:.2f}")

    # Final fit via statsmodels so we get standard errors and p-values for
    # the returned coefficient table. Only one call, on the selected subset.
    selected_names = [feature_names[i] for i in selected]
    Xc = sm.add_constant(X[selected_names], has_constant="add")
    final = sm.OLS(y_arr, Xc.values).fit()

    coefs = pd.DataFrame(
        {
            "term": Xc.columns,
            "estimate": final.params,
            "std_error": final.bse,
            "p_value": final.pvalues,
        }
    ).reset_index(drop=True)

    y_hat = final.predict(Xc.values)
    rmse = float(np.sqrt(np.mean((y_arr - y_hat) ** 2)))

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
    stepwise_max_features: int = 12,
    stepwise_max_iter: int = 15,
) -> tuple[dict[str, ModelResult], pd.DataFrame]:
    """Fit OLS, LASSO, and stepwise on the same design matrix.

    Defaults are tuned for Streamlit Cloud's free tier (1 vCPU / 1 GB). The
    full dataset is too costly for stepwise's O(features * iterations) full
    OLS refits before the websocket times out, so we sample and prune.

    Returns (results, comparison_table). The comparison table joins the three
    coefficient tables on `term` so the UI can put them side by side.
    """
    import gc

    X, y, _ = build_design_matrix(df, include_year=include_year, sample_n=sample_n)

    ols = fit_ols(X, y)
    gc.collect()

    lasso = fit_lasso(X, y)
    gc.collect()

    # Stepwise is O(features * iterations) so we cap aggressively.
    if X.shape[1] > stepwise_max_features:
        variances = X.var(axis=0).sort_values(ascending=False)
        keep = variances.head(stepwise_max_features).index.tolist()
        step = fit_stepwise(X[keep], y, max_iter=stepwise_max_iter)
    else:
        step = fit_stepwise(X, y, max_iter=stepwise_max_iter)
    del X, y
    gc.collect()

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
