"""
Boston 311 Service Requests dashboard (2015 - 2019).

Streamlit entry point. Run with:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis
import data_loader


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Boston 311 Dashboard",
    page_icon=":cityscape:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 4-color palette - keep it tight so the data does the talking.
PRIMARY = "#2F4858"      # dark slate, default data color
ACCENT = "#D85A30"       # coral, highlight / second series
MUTED = "#A8A29E"        # warm gray, tertiary / axis text
SURFACE = "#F2F0EA"      # cream, KPI card background

# Single-hue ramp built from PRIMARY -> ACCENT for sequential charts.
SEQUENTIAL_SCALE = [
    [0.00, "#E8E5DD"],
    [0.25, "#B8B6AE"],
    [0.50, "#7E8A8F"],
    [0.75, "#4D6571"],
    [1.00, PRIMARY],
]
# For categorical legends with up to 5 categories, stay within the palette.
CATEGORICAL_PALETTE = [PRIMARY, ACCENT, MUTED, "#7E8A8F", "#C68A66"]


CUSTOM_CSS = f"""
<style>
.block-container {{padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1280px;}}
.kpi-card {{background: {SURFACE}; border-radius: 10px; padding: 14px 16px;}}
.kpi-label {{font-size: 12px; color: {MUTED}; margin: 0 0 4px;}}
.kpi-value {{font-size: 26px; font-weight: 500; margin: 0; color: {PRIMARY};}}
.section-title {{font-size: 18px; font-weight: 500; margin: 12px 0 6px; color: {PRIMARY};}}
.muted {{color: {MUTED}; font-size: 13px;}}
hr {{margin: 12px 0;}}
/* Mute the multiselect chips so they read as data, not alerts. */
[data-baseweb="tag"] {{background-color: {SURFACE} !important; color: {PRIMARY} !important; border: 0.5px solid #D8D5CC !important;}}
[data-baseweb="tag"] svg {{fill: {PRIMARY} !important;}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_full_data() -> pd.DataFrame:
    return data_loader.load_data()


@st.cache_data(show_spinner=False)
def run_models_cached(year_lo: int, year_hi: int, include_year: bool) -> tuple[dict, pd.DataFrame]:
    """Slim, single-pass model fitting tuned for the free tier."""
    import gc

    full = load_full_data()
    needed = ["department", "neighborhood", "year", "resolution_hours"]
    df = full.loc[
        (full["year"] >= year_lo) & (full["year"] <= year_hi),
        needed,
    ].copy()
    if len(df) > 15_000:
        df = df.sample(n=15_000, random_state=42).reset_index(drop=True)
    gc.collect()

    results, comparison = analysis.run_all_models(
        df, include_year=include_year, sample_n=None
    )
    del df
    gc.collect()
    return results, comparison


def kpi_card(col, label: str, value: str) -> None:
    col.markdown(
        f"<div class='kpi-card'><p class='kpi-label'>{label}</p>"
        f"<p class='kpi-value'>{value}</p></div>",
        unsafe_allow_html=True,
    )


def fmt_int(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{int(n):,}"


def fmt_hours(h: float) -> str:
    if h >= 24:
        return f"{h / 24:.1f}d"
    return f"{h:.1f}h"


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

st.sidebar.title("Boston 311")
st.sidebar.caption("Service request resolution analysis, 2015 to 2019")

with st.sidebar:
    st.markdown("### Data")
    refresh = st.button("Refresh data cache", help="Re-download CSVs from data.boston.gov")
    if refresh:
        st.cache_data.clear()
        with st.spinner("Re-downloading raw CSVs..."):
            data_loader.load_data(force_refresh=True)
        st.success("Cache rebuilt")

try:
    with st.spinner("Loading 311 data (first run downloads ~1GB)..."):
        df_full = load_full_data()
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

available_years = sorted(int(y) for y in df_full["year"].dropna().unique())
available_depts = sorted(df_full["department"].dropna().unique().tolist())
available_nbhds = sorted(df_full["neighborhood"].dropna().unique().tolist())

with st.sidebar:
    st.markdown("### Filters")
    year_range = st.slider(
        "Year range",
        min_value=min(available_years),
        max_value=max(available_years),
        value=(min(available_years), max(available_years)),
    )
    selected_depts = st.multiselect(
        "Departments",
        options=available_depts,
        default=[],
        help="Empty = all departments",
    )
    selected_nbhds = st.multiselect(
        "Neighborhoods",
        options=available_nbhds,
        default=[],
        help="Empty = all neighborhoods",
    )
    cap_hours = st.number_input(
        "Cap resolution time (hours)",
        min_value=24,
        max_value=int(df_full["resolution_hours"].max()) + 1,
        value=5000,
        step=100,
        help="Trim extreme outliers from the long right tail",
    )

# Apply filters
mask = (df_full["year"] >= year_range[0]) & (df_full["year"] <= year_range[1])
if selected_depts:
    mask &= df_full["department"].isin(selected_depts)
if selected_nbhds:
    mask &= df_full["neighborhood"].isin(selected_nbhds)
mask &= df_full["resolution_hours"] <= cap_hours

df = df_full.loc[mask].copy()


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.markdown("## Boston 311 Service Requests, 2015 to 2019")
st.markdown(
    "<p class='muted'>Which factors influence case resolution time across departments, "
    "neighborhoods, and seasons</p>",
    unsafe_allow_html=True,
)

summary = data_loader.dataset_summary(df) if len(df) else {
    "total_cases": 0, "mean_hours": 0, "median_hours": 0, "slow_share": 0
}

k1, k2, k3, k4 = st.columns(4)
kpi_card(k1, "Total cases", fmt_int(summary["total_cases"]))
kpi_card(k2, "Mean resolution", fmt_hours(summary["mean_hours"]))
kpi_card(k3, "Median resolution", fmt_hours(summary["median_hours"]))
slowest_dept = (
    df.groupby("department")["resolution_hours"].mean().sort_values(ascending=False).index[0]
    if len(df) and df["department"].notna().any() else "n/a"
)
kpi_card(k4, "Slowest dept (mean)", str(slowest_dept))

st.markdown("---")

if len(df) == 0:
    st.warning("No rows match the current filters.")
    st.stop()


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab_overview, tab_dept, tab_nbhd, tab_time, tab_models = st.tabs(
    ["Overview", "Departments", "Neighborhoods", "Time trends", "Models"]
)


# -- Overview ----------------------------------------------------------------

with tab_overview:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<p class='section-title'>Resolution time distribution</p>", unsafe_allow_html=True)
        fig = px.histogram(
            df,
            x="resolution_hours",
            nbins=60,
            color_discrete_sequence=[PRIMARY],
        )
        fig.update_layout(
            bargap=0.02,
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            xaxis_title="Resolution hours",
            yaxis_title="Cases",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Heavily right-skewed: most cases resolve quickly, a small share runs for thousands of hours."
        )

    with c2:
        st.markdown("<p class='section-title'>Mean vs median by year</p>", unsafe_allow_html=True)
        agg = (
            df.groupby("year")["resolution_hours"]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=agg["year"], y=agg["mean"], mode="lines+markers",
            name="Mean", line=dict(color=PRIMARY, width=3),
        ))
        fig.add_trace(go.Scatter(
            x=agg["year"], y=agg["median"], mode="lines+markers",
            name="Median", line=dict(color=ACCENT, width=3),
        ))
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            xaxis=dict(dtick=1),
            yaxis_title="Hours",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Means drift up as a small group of very slow cases grows; medians actually fall."
        )

    st.markdown(
        "<p class='section-title'>Case volume by year (top 5 departments + other)</p>",
        unsafe_allow_html=True,
    )
    top5 = df["department"].value_counts().head(5).index.tolist()
    cat_df = df.copy()
    # Force to object so .where(..., "Other") works on pandas 3.0 even if the
    # column arrives as a CategoricalDtype.
    dept_obj = cat_df["department"].astype("object")
    cat_df["department_group"] = dept_obj.where(dept_obj.isin(top5), other="Other")
    pivot = (
        cat_df.groupby(["year", "department_group"]).size().reset_index(name="cases")
    )
    # Order categories so coloring runs from PRIMARY (top dept) to MUTED (other).
    order = top5 + ["Other"]
    fig = px.bar(
        pivot,
        x="year",
        y="cases",
        color="department_group",
        category_orders={"department_group": order},
        color_discrete_sequence=CATEGORICAL_PALETTE[: len(order)],
        barmode="stack",
        labels={"department_group": "Department"},
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=360,
        xaxis=dict(dtick=1),
        legend_title_text="",
    )
    st.plotly_chart(fig, width="stretch")


# -- Departments -------------------------------------------------------------

with tab_dept:
    st.markdown("<p class='section-title'>Mean resolution time by department (top 15)</p>", unsafe_allow_html=True)

    dept_stats = (
        df.groupby("department")["resolution_hours"]
        .agg(count="count", mean="mean", median="median", sd="std")
        .reset_index()
        .sort_values("mean", ascending=False)
        .head(15)
    )

    fig = px.bar(
        dept_stats.sort_values("mean"),
        x="mean",
        y="department",
        orientation="h",
        color="mean",
        color_continuous_scale=SEQUENTIAL_SCALE,
        labels={"mean": "Mean hours", "department": "Department"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=460, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    st.markdown("<p class='section-title'>Departmental detail</p>", unsafe_allow_html=True)
    display = dept_stats.copy()
    display["mean"] = display["mean"].round(1)
    display["median"] = display["median"].round(1)
    display["sd"] = display["sd"].round(1)
    display = display.rename(columns={
        "department": "Department", "count": "Cases",
        "mean": "Mean (h)", "median": "Median (h)", "sd": "SD (h)"
    })
    st.dataframe(display, width="stretch", hide_index=True)

    st.caption(
        "Means are inflated by long-tail outliers; the median column shows the typical case."
    )


# -- Neighborhoods -----------------------------------------------------------

with tab_nbhd:
    st.markdown("<p class='section-title'>Resolution time by neighborhood</p>", unsafe_allow_html=True)

    nbhd_stats = (
        df.dropna(subset=["neighborhood"])
        .groupby("neighborhood")["resolution_hours"]
        .agg(count="count", mean="mean", median="median")
        .reset_index()
    )
    min_cases = st.slider(
        "Minimum cases to include",
        min_value=0,
        max_value=int(nbhd_stats["count"].max()),
        value=500,
    )
    filtered = nbhd_stats[nbhd_stats["count"] >= min_cases].sort_values("mean", ascending=False)

    fig = px.bar(
        filtered.sort_values("mean"),
        x="mean",
        y="neighborhood",
        orientation="h",
        color="mean",
        color_continuous_scale=SEQUENTIAL_SCALE,
        labels={"mean": "Mean hours", "neighborhood": "Neighborhood"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=520, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    geo = df.dropna(subset=["latitude", "longitude"]).copy()
    if len(geo) > 5000:
        geo = geo.sample(5000, random_state=42)
    if len(geo):
        st.markdown(
            "<p class='section-title'>Sample of cases on the map (random 5K)</p>",
            unsafe_allow_html=True,
        )
        # st.map serializes through stdlib json which doesn't understand
        # numpy.float32; cast to native Python float64 first.
        geo_points = geo[["latitude", "longitude"]].dropna().astype("float64")
        st.map(geo_points, size=2)


# -- Time trends -------------------------------------------------------------

with tab_time:
    st.markdown("<p class='section-title'>Average resolution over time</p>", unsafe_allow_html=True)

    granularity = st.radio(
        "Granularity",
        options=["Year", "Month of year", "Day of week"],
        horizontal=True,
    )

    if granularity == "Year":
        agg = df.groupby("year")["resolution_hours"].mean().reset_index()
        fig = px.line(agg, x="year", y="resolution_hours", markers=True,
                      color_discrete_sequence=[PRIMARY])
        fig.update_layout(xaxis=dict(dtick=1), margin=dict(l=10, r=10, t=10, b=10), height=380)
    elif granularity == "Month of year":
        agg = df.groupby("month")["resolution_hours"].mean().reset_index()
        fig = px.line(agg, x="month", y="resolution_hours", markers=True,
                      color_discrete_sequence=[PRIMARY])
        fig.update_layout(xaxis=dict(dtick=1), margin=dict(l=10, r=10, t=10, b=10), height=380)
    else:
        agg = df.groupby("dayofweek")["resolution_hours"].mean().reset_index()
        agg["day"] = agg["dayofweek"].map(
            {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        )
        fig = px.bar(agg, x="day", y="resolution_hours", color_discrete_sequence=[PRIMARY])
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380)

    fig.update_yaxes(title="Mean resolution hours")
    st.plotly_chart(fig, width="stretch")

    st.markdown("<p class='section-title'>Weekly trend</p>", unsafe_allow_html=True)
    weekly = (
        df.assign(week_start=df["open_dt"].dt.to_period("W").dt.start_time)
        .groupby("week_start")["resolution_hours"].mean().reset_index()
    )
    fig = px.area(weekly, x="week_start", y="resolution_hours",
                  color_discrete_sequence=[PRIMARY])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320,
                      xaxis_title="Week", yaxis_title="Mean hours")
    st.plotly_chart(fig, width="stretch")


# -- Models ------------------------------------------------------------------

import json as _json
from pathlib import Path as _Path

PRECOMPUTED_PATH = _Path(__file__).resolve().parent / "data" / "processed" / "model_results.json"


@st.cache_data(show_spinner=False)
def load_precomputed_models() -> dict | None:
    """Return the pre-computed model results dict, or None if the file is absent."""
    if not PRECOMPUTED_PATH.exists():
        return None
    try:
        return _json.loads(PRECOMPUTED_PATH.read_text())
    except Exception:
        return None


def _render_models_from_payload(payload: dict) -> None:
    """Render the Models tab UI from a pre-computed JSON payload."""
    meta = payload.get("metadata", {})
    results = payload.get("results", {})
    comparison = pd.DataFrame(payload.get("comparison", []))

    st.caption(
        f"Fit on {meta.get('n_rows_used', '?'):,} cases, "
        f"sampled to {meta.get('sample_n', '?'):,} rows. "
        f"Computed {meta.get('computed_at', 'offline')}."
    )

    m1, m2, m3 = st.columns(3)
    for col, key in zip((m1, m2, m3), ("OLS", "LASSO", "Stepwise")):
        r = results.get(key)
        if not r:
            continue
        col.markdown(
            f"<div class='kpi-card'>"
            f"<p class='kpi-label'>{r['name']}</p>"
            f"<p class='kpi-value'>R&sup2; {r['r2']:.3f}</p>"
            f"<p class='muted'>RMSE {r['rmse']:.3f} &middot; {r['n_features']} features &middot; n={r['n_obs']:,}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<p class='section-title'>Coefficient comparison</p>", unsafe_allow_html=True)
    plot_df = comparison[comparison["term"] != "const"].copy()
    plot_df["max_abs"] = plot_df[["OLS", "LASSO", "Stepwise"]].abs().max(axis=1)
    plot_df = plot_df.sort_values("max_abs", ascending=False).head(20)

    long = plot_df.melt(
        id_vars="term",
        value_vars=["OLS", "LASSO", "Stepwise"],
        var_name="Model",
        value_name="Coefficient",
    )
    fig = px.bar(
        long,
        x="Coefficient",
        y="term",
        color="Model",
        barmode="group",
        color_discrete_map={"OLS": PRIMARY, "LASSO": ACCENT, "Stepwise": MUTED},
        orientation="h",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=620,
                      yaxis_title="", xaxis_title="Coefficient (log scale)")
    st.plotly_chart(fig, width="stretch")

    st.markdown("<p class='section-title'>Full coefficient table</p>", unsafe_allow_html=True)
    table = comparison.copy()
    for col in ("OLS", "LASSO", "Stepwise"):
        table[col] = table[col].round(3)
    st.dataframe(table, width="stretch", hide_index=True)


with tab_models:
    st.markdown("<p class='section-title'>OLS, LASSO, and Stepwise regression</p>", unsafe_allow_html=True)

    payload = load_precomputed_models()

    if payload is not None:
        st.markdown(
            "<p class='muted'>Predicting log(1 + resolution_hours) from department, "
            "neighborhood, and year. LASSO uses 5-fold cross-validated lambda; "
            "stepwise uses AIC. Results below were pre-computed offline on the "
            "full dataset and shipped with the app.</p>",
            unsafe_allow_html=True,
        )
        _render_models_from_payload(payload)
    else:
        st.markdown(
            "<p class='muted'>Pre-computed results file not found. Falling back "
            "to live fit on a 15K-row sample.</p>",
            unsafe_allow_html=True,
        )
        include_year = st.checkbox("Include year as a predictor", value=True)
        run = st.button("Fit models", type="primary")
        if run:
            with st.spinner("Fitting OLS, LASSO, and Stepwise (15K-row sample)..."):
                results, comparison = run_models_cached(year_range[0], year_range[1], include_year)
            m1, m2, m3 = st.columns(3)
            for col, key in zip((m1, m2, m3), ("OLS", "LASSO", "Stepwise")):
                r = results[key]
                col.markdown(
                    f"<div class='kpi-card'>"
                    f"<p class='kpi-label'>{r.name}</p>"
                    f"<p class='kpi-value'>R&sup2; {r.r2:.3f}</p>"
                    f"<p class='muted'>RMSE {r.rmse:.3f} &middot; {r.n_features} features &middot; n={r.n_obs:,}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("<p class='section-title'>Coefficient comparison</p>", unsafe_allow_html=True)
            plot_df = comparison[comparison["term"] != "const"].copy()
            plot_df["max_abs"] = plot_df[["OLS", "LASSO", "Stepwise"]].abs().max(axis=1)
            plot_df = plot_df.sort_values("max_abs", ascending=False).head(20)
            long = plot_df.melt(
                id_vars="term",
                value_vars=["OLS", "LASSO", "Stepwise"],
                var_name="Model",
                value_name="Coefficient",
            )
            fig = px.bar(long, x="Coefficient", y="term", color="Model", barmode="group",
                         color_discrete_map={"OLS": PRIMARY, "LASSO": ACCENT, "Stepwise": MUTED},
                         orientation="h")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=620,
                              yaxis_title="", xaxis_title="Coefficient (log scale)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info(
                "Click **Fit models** to run OLS, LASSO, and stepwise. "
                "On the free tier this can take a while - if it stalls, run "
                "`python precompute_models.py` locally and commit the JSON."
            )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Data source: data.boston.gov 311 Service Requests. "
    "Methodology mirrors the ALY6015 final project (Group 6, Feb 2026)."
)
