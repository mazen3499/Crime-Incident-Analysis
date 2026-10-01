<div align="center">

# 🛡️ Crime Incident Command Center
### End-to-End Crime Analytics — From Messy Raw Data to a Production-Grade Interactive Dashboard

*A full data-science pipeline — cleaning, feature engineering, unsupervised machine learning, and a live BI dashboard — built on a real-world-style crime-incidents dataset.*

[Overview](#-project-overview) • [Key Features](#-key-features) • [Data Pipeline](#-data-cleaning--engineering-pipeline) • [Key Insights](#-key-insights--findings) • [Tech Stack](#️-tech-stack) • [Getting Started](#-getting-started) • [Project Structure](#-project-structure)

---

</div>

## 📌 Project Overview
Raw operational data is rarely analysis-ready. This project takes an end-to-end dataset engineered with realistic real-world messiness — inconsistent spellings, mixed date formats, out-of-range values, duplicate records, and missing fields — and transforms it into:

1. **A fully auditable data-cleaning pipeline (Jupyter Notebook)**, documenting every technical decision from raw CSV to analysis-ready data.
2. **An interactive Streamlit dashboard — Crime Incident Command Center** — that allows an analyst or decision-maker to filter, explore, and drill into crime patterns in real time.
3. **An unsupervised machine learning layer (K-Means clustering)** that surfaces operational crime "profiles" the raw numbers alone don't show.

The goal: demonstrate a complete, professional data-science workflow — **Ingest → Clean → Engineer → Analyze → Visualize → Model** — mirroring a real public-safety analytics environment.

---

## ✨ Key Features

### 🖥️ The Dashboard (4 focused tabs, zero scrolling, fully filterable)

| Tab | What it answers |
| :--- | :--- |
| **📊 Overview** | What kinds of crime are most common? How severe are they? Which districts, weapons, and outcomes dominate? |
| **🕒 Time & Geography** | When do incidents spike — by year, month, hour, or day-of-week × hour heatmap? Where are the geographic hotspots? |
| **👥 Demographics & Financial** | Who's involved (age, gender, race of identified suspects) and what's the financial toll by crime type? |
| **🤖 Machine Learning (Clusters)** | What hidden incident "profiles" emerge when location, time, and financial impact are analyzed together via K-Means? |

- **Live sidebar filtering:** Year range, Crime Type, District, Severity, Case Status, and Weapon Used — every chart and KPI updates instantly, with a one-click *Reset Filters* button.
- **6 headline KPIs:** Total Incidents, Total Property Loss, Avg Loss/Incident, Open/Active Case Rate, Total Arrests, and Avg Suspect Age.
- **17+ purpose-built visualizations:** Combining Plotly Express and Graph Objects (bar charts, donut charts, heatmaps, and spatial density maps) with a consistent dark-mode UI.
- **Privacy by design:** Identifiable personal data (PII like names, phone numbers, and exact street addresses) is stripped out from the analytical/dashboard layer.

---

## 🧹 Data Cleaning & Engineering Pipeline

Every column was cleaned with a deliberate, defensible rule — not a blind `dropna()`:

| Challenge Found in Raw Data | Technique Applied |
| :--- | :--- |
| **Duplicate incidents** | De-duplicated on the primary key (`incident_id`). |
| **Inconsistent text casing & whitespace** | Vectorized strip → whitespace collapse → Title Case normalization. |
| **Fragmented category aliases** | Canonical alias mapping and **Fuzzy Matching (`thefuzz`)** per column (crime type, district, weapon, status). |
| **Mixed date/time formats** | Parsed via `pd.to_datetime(..., format="mixed")` with an explicit check for timestamp completeness. |
| **Out-of-range numeric values** | Valid range domain filtering combined with IQR-based outlier clipping. |
| **Invalid GPS coordinates** | Filtered within plausible geographic boundary ranges. |
| **Personally Identifiable Information (PII)** | Excluded from the dashboard layer to ensure privacy compliance. |

---

## 🔍 Key Insights & Findings
- **Financial Toll:** Over **$125M** in total recorded property loss, averaging ~$24,800 per incident.
- **Arrests & Outcomes:** Over **11,000** total arrests recorded across all incidents.
- **Active Cases:** Roughly 4 in 10 cases remain open, pending, or under active investigation at any given time.
- **Geographic Hotspots:** North and South districts report the highest incident volume among monitored districts — an ideal indicator for patrol reallocation.
- **Machine Learning Clustering:** Unsupervised K-Means clustering (using coordinates, time of day, and property damage) surfaced 4 distinct operational clusters.

---

## 🛠️ Tech Stack

| Layer | Tools |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Data Wrangling** | Pandas, NumPy, thefuzz |
| **Machine Learning** | scikit-learn (KMeans, StandardScaler) |
| **Notebook EDA** | Matplotlib, Seaborn, Missingno |
| **Dashboard** | Streamlit |
| **Visualization** | Plotly (Express & Graph Objects) |

---

## 📁 Project Structure

```text
Crime-Incident-Analysis/
├── Crime_Data after cleaning .ipynb   # Full cleaning, EDA, and K-Means notebook
├── crime3.py                          # Streamlit dashboard application
├── crime_incidents_cleaned.csv        # Final analysis-ready dataset
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
