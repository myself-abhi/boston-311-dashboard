# Boston 311 Service Requests Dashboard

Interactive Streamlit app exploring Boston's 311 non-emergency service
requests from 2015 to 2019. Built on top of the ALY6015 final project
(Group 6, Feb 2026) which asked: which factors influence case resolution
time?

## What's inside

- **Overview** - KPI cards, resolution-time histogram, mean vs median
  by year, case volume by department.
- **Departments** - mean / median / SD resolution time for the top 15
  departments.
- **Neighborhoods** - resolution time by neighborhood with a minimum
  case-count filter and a sample of cases on a map of Boston.
- **Time trends** - average resolution hours by year, month-of-year,
  day-of-week, and a weekly area chart.
- **Models** - fits OLS, LASSO (5-fold CV), and AIC stepwise regression
  on log(1 + resolution_hours) and compares coefficients side by side.

Sidebar filters control year range, department, neighborhood, and an
upper cap on resolution hours to handle the long-tail outliers.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On the first run the app downloads ~1GB of CSVs from
`data.boston.gov` (CKAN API, with hard-coded fallback URLs) into
`data/raw/`, cleans them, and writes a single parquet cache at
`data/boston_311_2015_2019.parquet`. Every subsequent run reads from
parquet in a few seconds.

To force a fresh download, use the "Refresh data cache" button in the
sidebar, or run:

```bash
python data_loader.py
```

## Project layout

```
boston-311-dashboard/
├── app.py              # Streamlit UI
├── data_loader.py      # Downloads + cleans + caches the 5 years of data
├── analysis.py         # OLS, LASSO (CV), and AIC stepwise regression
├── build_snapshot.py   # Builds the deploy-ready parquet snapshot
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml     # Theme overrides
└── data/
    ├── processed/      # Per-year parquet snapshot (committed to git)
    └── raw/            # Raw CSVs (gitignored)
```

## Deploy to GitHub + Streamlit Community Cloud

### One-time: build the snapshot

The app needs a per-year parquet snapshot under `data/processed/` so the
deployed version doesn't have to download 1 GB at startup. Run this once
locally after the initial data download has completed:

```bash
python build_snapshot.py
```

That writes `data/processed/boston_311_2015.parquet` through `..._2019.parquet`,
each around 10-30 MB after Snappy compression. They are committed to git;
the raw CSVs under `data/raw/` are not.

### Push to GitHub

```bash
cd boston-311-dashboard
git init
git add .
git commit -m "Initial commit: Boston 311 dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/boston-311-dashboard.git
git push -u origin main
```

If you prefer the GitHub CLI:

```bash
gh repo create boston-311-dashboard --public --source=. --push
```

### Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**.
3. Repository: `<your-username>/boston-311-dashboard`. Branch: `main`. Main file path: `app.py`.
4. Click **Deploy**. First build takes 2-3 minutes (Streamlit installs `requirements.txt`).
5. After deploy, the app URL is shareable: `https://<your-username>-boston-311-dashboard.streamlit.app`.

### Updating the deployed app

Any push to `main` redeploys automatically. To refresh the data, rebuild the
snapshot locally and commit:

```bash
python build_snapshot.py
git add data/processed/
git commit -m "Refresh 311 snapshot"
git push
```

## Notes on methodology

- The continuous outcome is `resolution_hours = closed_dt - open_dt`
  in hours. Regressions use `log(1 + resolution_hours)` to tame the
  heavy right tail.
- The binary slow-case flag (`slow_case`) marks any case whose
  resolution time exceeds the overall sample mean, matching the report.
- For tractability, regressions collapse categorical levels outside
  the top 15 departments and top 20 neighborhoods to "Other", and
  fit on a 120K-row stratified sample. You can change these caps in
  `analysis.build_design_matrix`.
- Stepwise selection walks forward and backward by AIC on a
  variance-pruned candidate set; results closely track the report's
  R/dplyr stepAIC output.

## Data source

[311 Service Requests on data.boston.gov](https://data.boston.gov/dataset/311-service-requests).
The CKAN package id is `311-service-requests`; this app pulls each
year's CSV resource by name match against `"2015"` ... `"2019"`.
