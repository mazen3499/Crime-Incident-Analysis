<div align="center">

# 🛡️ Crime Incident Command Center
### End-to-End Crime Analytics — From Messy Raw Data to a Production-Grade Interactive Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**A full data-science pipeline — cleaning, feature engineering, unsupervised machine learning, and a live BI dashboard — built on a real-world-style, deliberately "messy" crime-incidents dataset.**

[Overview](#-project-overview) •
[Features](#-key-features) •
[Data Pipeline](#-data-cleaning--engineering-pipeline) •
[Insights](#-key-insights--findings) •
[Getting Started](#-getting-started) •
[Tech Stack](#-tech-stack) •
[Roadmap](#-roadmap)

</div>

---

## 📌 Project Overview

Raw operational data is rarely analysis-ready. This project takes a **5,250-row crime-incidents dataset engineered with realistic real-world messiness** — inconsistent spellings, mixed date formats, out-of-range values, duplicate records, and missing fields — and turns it into:

1. A **fully auditable data-cleaning pipeline** (Jupyter Notebook), documenting every decision from raw CSV to analysis-ready data.
2. An **interactive, single-screen Streamlit dashboard** — *Crime Incident Command Center* — that lets an analyst or decision-maker filter, explore, and drill into crime patterns in real time.
3. An **unsupervised machine learning layer** (K-Means clustering) that surfaces crime "profiles" the raw numbers alone don't show.

The goal: demonstrate a complete, professional data-science workflow — **Ingest → Clean → Engineer → Analyze → Visualize → Model** — the same way it would be done for a real public-safety analytics team.

---

## ✨ Key Features

### 🖥️ The Dashboard (4 focused tabs, zero scrolling, fully filterable)

| Tab | What it answers |
|---|---|
| 📊 **Overview** | What kinds of crime are most common? How severe are they? Which districts, weapons, and outcomes dominate? |
| 🕒 **Time & Geography** | When do incidents spike — by year, month, hour, or day-of-week × hour combination? Where are the geographic hotspots? |
| 👥 **Demographics & Financial** | Who's involved (age, gender, race of identified suspects) and what's the financial toll by crime type? |
| 🤖 **Machine Learning (Clusters)** | What *hidden* incident "profiles" emerge when location, time, and financial impact are analyzed together via K-Means? |

- **Live sidebar filtering** — Year range, Crime Type, District, Severity, Case Status, and Weapon Used — every chart and KPI updates instantly, with a one-click **Reset Filters**.
- **7 headline KPIs** — Total Incidents, Total Property Loss, Avg Loss/Incident, Open/Active Case Rate, Total Arrests, Avg Suspect Age, and **Felony Rate** (powered by the engineered `crime_class` feature — see below).
- **17+ purpose-built visualizations** combining **Plotly Express** and **Plotly Graph Objects** (bar, donut, grouped bar, line, heatmap, and map-based charts), all sharing one consistent, accessible dark color system.
- **A dedicated Machine Learning tab** — not just a chart bolted onto the side, but a full tab breaking down what each K-Means cluster represents (geographic scatter, loss-vs-hour profile, and a per-cluster summary table).
- **Privacy by design** — names, phone numbers, and raw addresses are excluded from every visualization; only aggregate, non-identifying data is ever displayed.

### 🧪 The Notebook

- Step-by-step, heavily commented cleaning pipeline covering every column in the dataset.
- Before/after validation at every stage (`.info()`, `.isna().sum()`, `.value_counts()`, box plots, missingness matrices).
- Exploratory Data Analysis with 12+ Matplotlib/Seaborn visualizations.
- A dedicated **K-Means clustering** section for unsupervised pattern discovery.

---

## 🧹 Data Cleaning & Engineering Pipeline

Every column was cleaned with a deliberate, defensible rule — not a blind `dropna()`. Highlights:

| Challenge Found in Raw Data | Technique Applied |
|---|---|
| **200 duplicate incidents** (same `incident_id`) | De-duplicated on the primary key |
| **Inconsistent text casing & spacing** across every categorical column | Vectorized `strip → collapse whitespace → title-case` normalization applied uniformly |
| **Fragmented category aliases** (e.g. `"D.U.I."`, `"Dwi"`, `"Drunk Driving"` all meaning `DUI`) | Canonical alias-mapping dictionaries per column (crime type, district, gender, weapon, case status, resolution, severity), engineered by auditing the *actual* unique values rather than guessing |
| **Mixed date/time formats, and date-only timestamps with no clock time** | `pd.to_datetime(..., format="mixed")` with an explicit check for the presence of a time component — preventing a spurious "midnight spike" artifact in hour-of-day analysis |
| **Out-of-range and negative numeric values** (ages, arrest counts, property loss) | Domain-valid range checks (e.g. age ∈ [0, 100]) combined with IQR-based outlier clipping |
| **Invalid GPS coordinates** | Bounded to plausible U.S. latitude/longitude ranges before any geographic analysis |
| **Personally identifiable information** (names, phone numbers, raw street addresses) | Deliberately excluded from the analytical/dashboard layer — a security-by-design decision, not an oversight |

#### 🏛️ Engineered Feature: Felony / Misdemeanor Classification

Beyond cleaning, a `crime_class` feature was engineered from scratch to turn raw `crime_type` labels into an operationally meaningful legal severity tier:

- **Clear-cut crime types** (e.g. *Homicide, Robbery, Burglary, Arson, Drug Offense*) are classified directly as **Felony**; low-harm types (e.g. *Vandalism, Graffiti, Trespassing*) as **Misdemeanor**.
- **Context-dependent crime types** — where the real-world charge genuinely depends on circumstances — are resolved with an objective, auditable rule instead of a guess:
  - *Assault, Battery, Domestic Violence, DUI* → classified using the incident's recorded **severity** (High/Critical → Felony).
  - *Theft, Larceny, Fraud, Scam* and similar → classified using a **$2,500 property-loss threshold**, mirroring the petty/grand-theft line used in many U.S. jurisdictions.
- Every incident resolved via one of these thresholds (rather than a direct crime-type match) is flagged with `crime_class_needs_review = True` — surfacing exactly the ~38% of cases a human analyst should sanity-check, instead of presenting the entire classification as equally certain.

**Final cleaned dataset:** `crime_incidents_cleaned.csv` — **5,050 incidents × 41 engineered columns**, including derived fields such as `incident_hour`, `incident_dow`, `crime_class`, `crime_class_needs_review`, and K-Means cluster assignments.

---

## 🔍 Key Insights & Findings

> Figures below are computed directly from the final `crime_incidents_cleaned.csv` (5,050 incidents, 2018–2024).

- 💰 **$125.7M** in total recorded property loss, averaging **~$24,888 per incident**.
- 🚔 **11,731 total arrests** recorded across all incidents.
- ⚖️ **70.7% of incidents are classified as Felony-tier**, based on the engineered `crime_class` feature — with ~38% of all classifications flagged for human review (see [Data Cleaning](#-data-cleaning--engineering-pipeline)).
- 🔺 **Assault, Trespassing, Drug Offense, Robbery,** and **Arson** are the five most frequently reported crime types.
- 🗺️ **North** is the highest-volume district (696 incidents) among the 10 monitored districts — a clear candidate for resource reallocation analysis.
- 📈 Year-over-year incident volume is **broadly stable** (no runaway upward trend across 2018–2024), which is itself an actionable finding: any future spike would be immediately visible against this baseline.
- 🧭 Roughly **65% of cases** remain open, pending, or under active investigation at any given time.
- 🤖 **K-Means clustering** (on location, hour, and financial loss) surfaces distinct incident "profiles" invisible in single-variable breakdowns — e.g., clusters combining specific geographic zones with characteristic time-of-day and loss patterns, useful for targeted patrol planning.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data Wrangling** | Pandas, NumPy |
| **Machine Learning** | scikit-learn (`KMeans`, `StandardScaler`) |
| **Notebook EDA** | Matplotlib, Seaborn, `missingno` |
| **Interactive Dashboard** | Streamlit |
| **Visualization Engine** | Plotly (Express + Graph Objects) |
| **Environment** | Jupyter Notebook |

---

## 📁 Project Structure

```
crime-incident-command-center/
├── Crime_Data_after_cleaning.ipynb    # Full data cleaning + EDA + K-Means notebook (pre-executed)
├── crime3.py                          # Streamlit dashboard application
├── crime_incidents_cleaned.csv        # Final, analysis-ready dataset (41 columns)
├── crime_incidents_messy.csv          # Original raw dataset (for reproducibility)
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md                          # You are here
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/mazen3499/crime-incident-command-center.git
cd crime-incident-command-center

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

Make sure `crime_incidents_cleaned.csv` (or `crime_incidents_messy.csv`) is in the same folder as `crime3.py`, then:

```bash
streamlit run crime3.py
```

The dashboard opens automatically at `http://localhost:8501`.

### Explore the Notebook

The notebook is shipped **already executed** — every cell's output, including all EDA charts and the clustering results, renders immediately on GitHub or when opened locally, with no re-run required:

```bash
jupyter notebook Crime_Data_after_cleaning.ipynb
```

---

## 🗺️ Roadmap

- [ ] Descriptive, human-readable labels for each K-Means cluster (e.g., *"Late-Night High-Value Downtown Incidents"*) instead of numeric IDs
- [ ] Automated PDF/Excel export of the filtered dashboard view
- [ ] Time-series forecasting (e.g., Prophet/ARIMA) for next-period incident volume
- [ ] Deploy to Streamlit Community Cloud for a live public demo link
- [ ] Automated data-quality regression tests (`pytest`) for the cleaning pipeline

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — feel free to fork, adapt, and build on it.

## 🙌 Acknowledgments

Built as part of a hands-on data analytics training project, applying a complete real-world pipeline — from raw, messy data to a decision-ready interactive product.

---

<div align="center">

**If this project was useful or interesting, consider giving it a ⭐ — it helps a lot!**

</div>
