from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

DATA_FILENAME = "crime_incidents_cleaned2.csv"
DATA_FILENAME_FALLBACK = "crime_incidents_cleaned.csv"
DATA_FILENAME_RAW = "crime_incidents_messy.csv"

BG = "#0B1220"
PANEL = "#131B2E"
BORDER = "#22304A"
TEXT = "#E7ECF5"
TEXT_MUTED = "#8592AC"

ACCENT_AMBER = "#F2A93B"
ACCENT_CYAN = "#33C2FF"
ACCENT_VIOLET = "#A78BFA"
ACCENT_GREEN = "#34D399"
ACCENT_RED = "#F0454B"
ACCENT_ORANGE = "#FB923C"
ACCENT_BLUE = "#60A5FA"
ACCENT_PINK = "#F472B6"

SEVERITY_ORDER = ["Low", "Medium", "High", "Critical", "Unknown"]
SEVERITY_COLORS = {
    "Low": ACCENT_GREEN,
    "Medium": "#FBBF24",
    "High": ACCENT_ORANGE,
    "Critical": ACCENT_RED,
    "Unknown": TEXT_MUTED,
}

QUALITATIVE_PALETTE = [
    ACCENT_CYAN,
    ACCENT_AMBER,
    ACCENT_VIOLET,
    ACCENT_GREEN,
    ACCENT_RED,
    ACCENT_ORANGE,
    ACCENT_BLUE,
    ACCENT_PINK,
    "#4ADE80",
    "#38BDF8",
    "#FCA5A5",
    "#C084FC",
]

FONT_FAMILY = "Inter, -apple-system, sans-serif"
HEADER_FONT_FAMILY = "'Space Grotesk', Inter, sans-serif"

CHART_H = 205
CHART_H_LARGE = 245

DOW_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

CRIME_TYPE_MAPPING = {
    "Arsen": "Arson",
    "Asslt": "Assault",
    "B&E": "Breaking & Entering",
    "Burglry": "Burglary",
    "Cyber Crime": "Cybercrime",
    "D.U.I.": "DUI",
    "Dui": "DUI",
    "Duii": "DUI",
    "Dwi": "DUI",
    "Drunk Driving": "DUI",
    "Dom. Violence": "Domestic Violence",
    "Domestc Violence": "Domestic Violence",
    "Drug Offence": "Drug Offense",
    "Drugs": "Drug Offense",
    "Narcotics": "Drug Offense",
    "Dv": "Domestic Violence",
    "Homocide": "Homicide",
    "Kidnaping": "Kidnapping",
    "Robbry": "Robbery",
    "Roberry": "Robbery",
    "Sa": "Sexual Assault",
    "Sex Assault": "Sexual Assault",
    "Sexual Assualt": "Sexual Assault",
    "Tresspassing": "Trespassing",
    "Trespass": "Trespassing",
    "Vandlism": "Vandalism",
    "Fire Setting": "Arson",
    "Deception": "Fraud",
    "Fraudulent Activity": "Fraud",
    "Online Fraud": "Fraud",
    "Scam": "Fraud",
    "Murder": "Homicide",
    "Manslaughter": "Homicide",
    "Larceny": "Theft/Larceny",
    "Theft": "Theft/Larceny",
    "Stealing": "Theft/Larceny",
}
DISTRICT_MAPPING = {
    "Cen": "Central",
    "Eas": "East",
    "Mid": "Midtown",
    "Nor": "North",
    "Sou": "South",
    "Wes": "West",
}
GENDER_MAPPING = {"F": "Female", "M": "Male", "Unknow": "Unknown"}
VICTIM_GENDER_MAPPING = {
    "M": "Male",
    "F": "Female",
    "Other": "Other",
    "Unknown": "Unknown",
}
CASE_STATUS_MAPPING = {
    "Pendng": "Pending",
    "Investgation": "Under Investigation",
}
WEAPON_MAPPING = {
    "Gun": "Firearm",
    "Pistol": "Firearm",
    "Rifle": "Firearm",
    "Hands": "Hands / Unarmed",
    "Hands/Feet": "Hands / Unarmed",
    "Unarmed": "Hands / Unarmed",
    "Bat": "Blunt Object",
}
RESOLUTION_MAPPING = {
    "Arres Made": "Arrest Made",
    "Warning Issued": "Warning",
    "Case Dismissed": "Dismissed",
}
SEVERITY_MAPPING = {
    "1": "Low",
    "2": "Medium",
    "3": "High",
    "4": "Critical",
    "med": "Medium",
    "Med": "Medium",
    "medium": "Medium",
    "low": "Low",
    "high": "High",
    "Medium": "Medium",
    "crit": "Critical",
    "Crit": "Critical",
    "criticl": "Critical",
    "critical": "Critical",
}
REPORTED_ONLINE_MAPPING = {
    "True": True,
    "Yes": True,
    "1": True,
    "False": False,
    "No": False,
    "0": False,
}

PII_COLUMNS_TO_DROP = [
    "officer_first_name",
    "officer_last_name",
    "suspect_first_name",
    "suspect_last_name",
    "victim_first_name",
    "victim_last_name",
    "victim_phone",
    "address",
    "notes",
]


def inject_custom_css() -> None:
  st.markdown(
      f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: {FONT_FAMILY};
        }}

        .stApp {{
            background: {BG};
        }}

        .block-container {{
            padding-top: 0.5rem;
            padding-bottom: 0.3rem;
            padding-left: 1.4rem;
            padding-right: 1.4rem;
            max-width: 100%;
        }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        div[data-testid="stHeader"] {{ background: transparent; height: 0.4rem; }}
        div[data-testid="stDecoration"] {{ display: none; }}
        div[data-testid="stVerticalBlock"] {{ gap: 0.5rem; }}
        div[data-testid="stElementContainer"] {{ margin-bottom: 0; }}

        section[data-testid="stSidebar"] {{
            background: {PANEL};
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.2rem;
        }}

        h1, h2, h3, .dash-title {{
            font-family: {HEADER_FONT_FAMILY};
            color: {TEXT};
        }}
        p, span, label, div {{
            color: {TEXT};
        }}

        .dash-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.2rem 0 0.4rem 0;
            border-bottom: 1px solid {BORDER};
            margin-bottom: 0.5rem;
        }}
        .dash-title {{
            font-size: 1.35rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: 0.2px;
        }}
        .dash-subtitle {{
            font-size: 0.78rem;
            color: {TEXT_MUTED};
            margin-top: 0.1rem;
        }}

        .kpi-card {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-top: 3px solid var(--kpi-accent, {ACCENT_CYAN});
            border-radius: 10px;
            padding: 0.55rem 0.8rem;
            min-height: 84px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.1rem;
        }}
        .kpi-icon {{ font-size: 1.05rem; opacity: 0.9; line-height: 1; }}
        .kpi-value {{
            font-family: {HEADER_FONT_FAMILY};
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.15;
            color: {TEXT};
            white-space: nowrap;
        }}
        .kpi-label {{
            font-size: 0.68rem;
            color: {TEXT_MUTED};
            text-transform: none;
            letter-spacing: 0.2px;
            line-height: 1.2;
        }}

        div[class*="st-key-chart_card_"] {{
            background: {PANEL} !important;
            border: 1px solid {BORDER} !important;
            border-radius: 10px !important;
            padding: 0.5rem 0.7rem 0.35rem 0.7rem !important;
        }}
        .chart-card-title {{
            font-size: 0.82rem;
            font-weight: 600;
            color: {TEXT};
            margin-bottom: 0.2rem;
        }}

        button[data-baseweb="tab"] {{
            font-family: {HEADER_FONT_FAMILY};
            font-size: 0.9rem;
            color: {TEXT_MUTED};
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {ACCENT_AMBER};
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: {ACCENT_AMBER};
        }}
        div[data-baseweb="tab-border"] {{
            background-color: {BORDER};
        }}

        div.stButton > button {{
            background: {PANEL};
            border: 1px solid {ACCENT_AMBER};
            color: {ACCENT_AMBER};
            border-radius: 8px;
            font-size: 0.8rem;
            padding: 0.3rem 0.8rem;
        }}
        div.stButton > button:hover {{
            background: {ACCENT_AMBER};
            color: {BG};
            border: 1px solid {ACCENT_AMBER};
        }}

        span[data-baseweb="tag"] {{
            background-color: {ACCENT_AMBER} !important;
            color: {BG} !important;
        }}
        </style>
        """,
      unsafe_allow_html=True,
  )


def load_raw_data(uploaded_file) -> Optional[pd.DataFrame]:
  try:
    if uploaded_file is not None:
      return pd.read_csv(uploaded_file)
    if os.path.exists(DATA_FILENAME):
      return pd.read_csv(DATA_FILENAME)
    if os.path.exists(DATA_FILENAME_FALLBACK):
      return pd.read_csv(DATA_FILENAME_FALLBACK)
    if os.path.exists(DATA_FILENAME_RAW):
      return pd.read_csv(DATA_FILENAME_RAW)
  except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
  return None


def _normalize_key(value) -> str:
  return " ".join(str(value).strip().lower().split())


def apply_alias_mapping(series: pd.Series, mapping: dict) -> pd.Series:
  normalized_map = {_normalize_key(k): v for k, v in mapping.items()}

  def _resolve(value):
    if pd.isna(value):
      return value
    return normalized_map.get(_normalize_key(value), value)

  return series.apply(_resolve)


def _clip_outliers_iqr(series: pd.Series) -> pd.Series:
  q1, q3 = series.quantile(0.25), series.quantile(0.75)
  iqr = q3 - q1
  lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
  return series.clip(lower=lower, upper=upper)


@st.cache_data(show_spinner="Preparing cleaned dataset for dashboard…")
def clean_crime_data(df_raw: pd.DataFrame) -> pd.DataFrame:
  df = df_raw.copy()

  if "latitude_per_state" in df.columns and "latitude" not in df.columns:
    df.rename(
        columns={
            "latitude_per_state": "latitude",
            "longitude_per_state": "longitude",
        },
        inplace=True,
    )

  df.drop_duplicates(subset="incident_id", inplace=True)

  cat_cols = df.select_dtypes(include="object").columns
  df[cat_cols] = df[cat_cols].apply(
      lambda x: (
          x.str.strip().str.replace(r"\s+", " ", regex=True).str.lower().str.title()
      )
  )

  df["crime_type"] = apply_alias_mapping(df["crime_type"], CRIME_TYPE_MAPPING)
  df["district"] = apply_alias_mapping(df["district"], DISTRICT_MAPPING)
  df["suspect_gender"] = apply_alias_mapping(
      df["suspect_gender"], GENDER_MAPPING
  ).fillna("Unknown")
  df["victim_gender"] = apply_alias_mapping(
      df["victim_gender"], VICTIM_GENDER_MAPPING
  )
  df["weapon_used"] = apply_alias_mapping(df["weapon_used"], WEAPON_MAPPING)
  df["resolution"] = apply_alias_mapping(df["resolution"], RESOLUTION_MAPPING)
  df["case_status"] = apply_alias_mapping(
      df["case_status"], CASE_STATUS_MAPPING
  )
  df["severity"] = apply_alias_mapping(
      df["severity"], SEVERITY_MAPPING
  ).fillna("Unknown")

  fill_unknown_cols = [
      "district",
      "crime_type",
      "severity",
      "case_status",
      "weapon_used",
      "suspect_gender",
      "victim_gender",
      "victim_id",
      "resolution",
  ]
  for col in fill_unknown_cols:
    if col in df.columns:
      df[col] = df[col].fillna("Unknown")

  if "reported_online" in df.columns:
    df["reported_online"] = df["reported_online"].replace(
        REPORTED_ONLINE_MAPPING
    )
    df["reported_online"] = df["reported_online"].apply(
        lambda v: v if isinstance(v, bool) else ("Unknown" if pd.isna(v) else v)
    )

  if "latitude" in df.columns:
    df["latitude"] = df["latitude"].where(df["latitude"].between(24, 49))
  if "longitude" in df.columns:
    df["longitude"] = df["longitude"].where(df["longitude"].between(-125, -67))

  if "incident_datetime" in df.columns:
    raw_dt_str = df["incident_datetime"].astype(str)
    has_time_component = raw_dt_str.str.contains(":", na=False)
    df["incident_datetime"] = pd.to_datetime(
        df["incident_datetime"], format="mixed", errors="coerce"
    )
    if "incident_hour" not in df.columns or df["incident_hour"].isna().all():
      df["incident_hour"] = df["incident_datetime"].dt.hour.where(
          has_time_component
      )
    if "incident_year" not in df.columns or df["incident_year"].isna().all():
      df["incident_year"] = df["incident_datetime"].dt.year
    if "incident_month" not in df.columns or df["incident_month"].isna().all():
      df["incident_month"] = df["incident_datetime"].dt.month
    if "incident_dow" not in df.columns or df["incident_dow"].isna().all():
      df["incident_dow"] = df["incident_datetime"].dt.day_name()

  for age_col in ["suspect_age", "victim_age"]:
    if age_col in df.columns:
      df[age_col] = df[age_col].where(df[age_col].between(0, 100))
      df[age_col] = _clip_outliers_iqr(df[age_col])

  if "num_arrests" in df.columns:
    df["num_arrests"] = (
        pd.to_numeric(df["num_arrests"], errors="coerce").abs().astype("Int64")
    )
  if "property_loss_usd" in df.columns:
    df["property_loss_usd"] = pd.to_numeric(
        df["property_loss_usd"], errors="coerce"
    ).abs()
    df["property_loss_usd"] = _clip_outliers_iqr(
        df["property_loss_usd"]
    ).fillna(0.0)

  df["hour_for_cluster"] = df["incident_hour"].fillna(
      df["incident_hour"].median()
  )
  features = ["latitude", "longitude", "hour_for_cluster", "property_loss_usd"]
  valid_cluster_idx = df[features].dropna().index

  if len(valid_cluster_idx) > 10:
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(df.loc[valid_cluster_idx, features])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df.loc[valid_cluster_idx, "crime_cluster"] = kmeans.fit_predict(
        scaled_feats
    ).astype(str)
  else:
    df["crime_cluster"] = "Unassigned"

  df.drop(
      columns=[c for c in PII_COLUMNS_TO_DROP if c in df.columns],
      inplace=True,
      errors="ignore",
  )
  return df


FILTER_KEYS = [
    "f_year_range",
    "f_crime_type",
    "f_district",
    "f_severity",
    "f_case_status",
    "f_weapon",
]


def _reset_filters() -> None:
  for key in FILTER_KEYS:
    st.session_state.pop(key, None)


def render_sidebar_filters(df: pd.DataFrame) -> dict:
  with st.sidebar:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.2rem;'>"
        f"<span style='font-size:1.3rem;'>🛡️</span>"
        f"<span style='font-family:{HEADER_FONT_FAMILY};font-size:1.05rem;font-weight:700;'>"
        f"CRIME ANALYTICS</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Filters apply to every tab")
    st.button("🔄 Reset Filters", on_click=_reset_filters, width="stretch")
    st.markdown("---")

    years = df["incident_year"].dropna()
    year_min, year_max = (
        (int(years.min()), int(years.max())) if not years.empty else (2018, 2024)
    )

    st.markdown("**📅 Year Range**")
    year_range = st.slider(
        "Year range",
        year_min,
        year_max,
        (year_min, year_max),
        key="f_year_range",
        label_visibility="collapsed",
    )

    st.markdown("**🗂️ Crime Type**")
    crime_types = sorted(df["crime_type"].dropna().unique())
    sel_crime = st.multiselect(
        "Crime type",
        crime_types,
        default=crime_types,
        key="f_crime_type",
        label_visibility="collapsed",
    )

    st.markdown("**📍 District**")
    districts = sorted(df["district"].dropna().unique())
    sel_district = st.multiselect(
        "District",
        districts,
        default=districts,
        key="f_district",
        label_visibility="collapsed",
    )

    st.markdown("**⚠️ Severity**")
    severities = [s for s in SEVERITY_ORDER if s in df["severity"].unique()]
    sel_severity = st.multiselect(
        "Severity",
        severities,
        default=severities,
        key="f_severity",
        label_visibility="collapsed",
    )

    st.markdown("**📁 Case Status**")
    statuses = sorted(df["case_status"].dropna().unique())
    sel_status = st.multiselect(
        "Case status",
        statuses,
        default=statuses,
        key="f_case_status",
        label_visibility="collapsed",
    )

    st.markdown("**🔫 Weapon Used**")
    weapons = sorted(df["weapon_used"].dropna().unique())
    sel_weapon = st.multiselect(
        "Weapon used",
        weapons,
        default=weapons,
        key="f_weapon",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Operational Crime Analytics Dashboard")

  return {
      "year_range": year_range,
      "crime_type": sel_crime,
      "district": sel_district,
      "severity": sel_severity,
      "case_status": sel_status,
      "weapon_used": sel_weapon,
  }


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
  mask = (
      df["crime_type"].isin(f["crime_type"])
      & df["district"].isin(f["district"])
      & df["severity"].isin(f["severity"])
      & df["case_status"].isin(f["case_status"])
      & df["weapon_used"].isin(f["weapon_used"])
      & (
          df["incident_year"].between(f["year_range"][0], f["year_range"][1])
          | df["incident_year"].isna()
      )
  )
  return df.loc[mask]


def format_compact_number(
    value: float, prefix: str = "", decimals: int = 2
) -> str:
  if pd.isna(value):
    return "—"
  abs_v = abs(value)
  if abs_v >= 1_000_000:
    return f"{prefix}{value / 1_000_000:.{decimals}f}M"
  if abs_v >= 1_000:
    return f"{prefix}{value / 1_000:.{decimals}f}K"
  return f"{prefix}{value:,.{0 if float(value).is_integer() else decimals}f}"


def compute_kpis(df: pd.DataFrame) -> dict:
  total_incidents = len(df)
  total_loss = df["property_loss_usd"].sum()
  avg_loss = df["property_loss_usd"].mean() if total_incidents else 0
  open_like = (
      df["case_status"].isin(["Open", "Pending", "Under Investigation"]).sum()
  )
  open_pct = (open_like / total_incidents * 100) if total_incidents else 0
  total_arrests = (
      df["num_arrests"].sum() if "num_arrests" in df.columns else 0
  )
  total_arrests = 0 if pd.isna(total_arrests) else int(total_arrests)
  avg_suspect_age = (
      df["suspect_age"].mean()
      if "suspect_age" in df.columns and total_incidents
      else 0
  )
  return {
      "total_incidents": total_incidents,
      "total_loss": total_loss,
      "avg_loss": avg_loss,
      "open_pct": open_pct,
      "total_arrests": total_arrests,
      "avg_suspect_age": avg_suspect_age,
  }


def _kpi_card_html(icon: str, label: str, value: str, accent: str) -> str:
  return f"""
    <div class="kpi-card" style="--kpi-accent:{accent};">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """


def render_kpi_row(kpis: dict) -> None:
  cols = st.columns(6, gap="small")
  cards = [
      ("🧾", "Total Incidents", f"{kpis['total_incidents']:,}", ACCENT_CYAN),
      (
          "💰",
          "Total Property Loss",
          format_compact_number(kpis["total_loss"], prefix="$"),
          ACCENT_AMBER,
      ),
      (
          "📉",
          "Avg Loss / Incident",
          format_compact_number(kpis["avg_loss"], prefix="$"),
          ACCENT_VIOLET,
      ),
      ("🕵️", "Open / Active Cases", f"{kpis['open_pct']:.1f}%", ACCENT_ORANGE),
      ("🚔", "Total Arrests", f"{int(kpis['total_arrests']):,}", ACCENT_GREEN),
      (
          "🧑",
          "Avg Suspect Age",
          (
              f"{kpis['avg_suspect_age']:.1f}"
              if pd.notna(kpis["avg_suspect_age"])
              else "—"
          ),
          ACCENT_BLUE,
      ),
  ]
  for col, (icon, label, value, accent) in zip(cols, cards):
    with col:
      st.markdown(
          _kpi_card_html(icon, label, value, accent), unsafe_allow_html=True
      )


STATIC_CHART_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
    "doubleClick": False,
    "showTips": False,
}


def _style_fig(
    fig: go.Figure, height: int, show_legend: bool = False
) -> go.Figure:
  fig.update_layout(
      height=height,
      margin=dict(l=8, r=28, t=8, b=8),
      paper_bgcolor=PANEL,
      plot_bgcolor=PANEL,
      font=dict(family=FONT_FAMILY, color=TEXT, size=11),
      showlegend=show_legend,
      legend=dict(
          orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=9)
      ),
      hoverlabel=dict(bgcolor=BG, font_size=11, font_family=FONT_FAMILY),
      dragmode=False,
  )
  fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
  fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
  return fig


def _slugify(text: str) -> str:
  return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def chart_card(title: str, fig: go.Figure) -> None:
  with st.container(key=f"chart_card_{_slugify(title)}"):
    st.markdown(
        f'<div class="chart-card-title">{title}</div>', unsafe_allow_html=True
    )
    st.plotly_chart(fig, width="stretch", config=STATIC_CHART_CONFIG)


def fig_crime_type_bar(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
  counts = df["crime_type"].value_counts().head(top_n).sort_values()
  fig = go.Figure(
      go.Bar(
          x=counts.values,
          y=counts.index,
          orientation="h",
          marker=dict(color=ACCENT_CYAN),
          text=counts.values,
          textposition="auto",
          textfont=dict(size=9),
      )
  )
  return _style_fig(fig, CHART_H)


def fig_severity_donut(df: pd.DataFrame) -> go.Figure:
  counts = df["severity"].value_counts()
  order = [s for s in SEVERITY_ORDER if s in counts.index]
  counts = counts.reindex(order)
  fig = go.Figure(
      go.Pie(
          labels=counts.index,
          values=counts.values,
          hole=0.55,
          marker=dict(colors=[SEVERITY_COLORS[s] for s in counts.index]),
          textinfo="percent",
          textfont=dict(size=10),
      )
  )
  return _style_fig(fig, CHART_H, show_legend=True)


def fig_district_bar(df: pd.DataFrame) -> go.Figure:
  counts = df["district"].value_counts().sort_values(ascending=False)
  fig = go.Figure(
      go.Bar(
          x=counts.index,
          y=counts.values,
          marker=dict(color=ACCENT_AMBER),
          text=counts.values,
          textposition="auto",
          textfont=dict(size=9),
      )
  )
  return _style_fig(fig, CHART_H)


def fig_case_status_resolution(df: pd.DataFrame) -> go.Figure:
  status_counts = df["case_status"].value_counts()
  resolution_counts = df["resolution"].value_counts()
  fig = make_subplots(
      rows=1, cols=2, subplot_titles=("Case Status", "Resolution")
  )
  fig.add_trace(
      go.Bar(
          x=status_counts.index,
          y=status_counts.values,
          marker=dict(color=ACCENT_VIOLET),
      ),
      row=1,
      col=1,
  )
  fig.add_trace(
      go.Bar(
          x=resolution_counts.index,
          y=resolution_counts.values,
          marker=dict(color=ACCENT_PINK),
      ),
      row=1,
      col=2,
  )
  fig.update_annotations(font=dict(size=10, color=TEXT_MUTED))
  fig = _style_fig(fig, CHART_H)
  fig.update_layout(margin=dict(t=26))
  return fig


def fig_weapon_bar(df: pd.DataFrame) -> go.Figure:
  counts = (
      df[df["weapon_used"] != "Unknown"]["weapon_used"]
      .value_counts()
      .sort_values()
  )
  fig = go.Figure(
      go.Bar(
          x=counts.values,
          y=counts.index,
          orientation="h",
          marker=dict(color=ACCENT_GREEN),
          text=counts.values,
          textposition="auto",
          textfont=dict(size=9),
      )
  )
  return _style_fig(fig, CHART_H)


def fig_gender_comparison(df: pd.DataFrame) -> go.Figure:
  suspect_counts = df["suspect_gender"].value_counts()
  victim_counts = df["victim_gender"].value_counts()
  genders = sorted(set(suspect_counts.index) | set(victim_counts.index))
  fig = go.Figure()
  fig.add_trace(
      go.Bar(
          name="Suspect",
          x=genders,
          y=[suspect_counts.get(g, 0) for g in genders],
          marker=dict(color=ACCENT_RED),
      )
  )
  fig.add_trace(
      go.Bar(
          name="Victim",
          x=genders,
          y=[victim_counts.get(g, 0) for g in genders],
          marker=dict(color=ACCENT_BLUE),
      )
  )
  fig.update_layout(barmode="group")
  return _style_fig(fig, CHART_H, show_legend=True)


def fig_yearly_trend(df: pd.DataFrame) -> go.Figure:
  yearly = df["incident_year"].dropna().value_counts().sort_index()
  fig = go.Figure(
      go.Scatter(
          x=yearly.index,
          y=yearly.values,
          mode="lines+markers+text",
          text=[f"{int(v)}" for v in yearly.values],
          textposition="top center",
          textfont=dict(size=10, color=TEXT),
          line=dict(color=ACCENT_CYAN, width=2.5),
          marker=dict(size=7),
          fill="tozeroy",
          fillcolor="rgba(51,194,255,0.12)",
      )
  )
  fig = _style_fig(fig, CHART_H)
  max_val = yearly.values.max() if not yearly.empty else 100
  fig.update_layout(
      yaxis=dict(range=[0, max_val * 1.25]),
      margin=dict(t=38),
  )
  return fig


def fig_monthly_bar(df: pd.DataFrame) -> go.Figure:
  monthly = df["incident_month"].dropna().value_counts().sort_index()
  month_labels = [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec",
  ]
  fig = go.Figure(
      go.Bar(
          x=[month_labels[int(m) - 1] for m in monthly.index],
          y=monthly.values,
          marker=dict(color=ACCENT_AMBER),
      )
  )
  return _style_fig(fig, CHART_H)


def fig_hourly_bar(df: pd.DataFrame) -> go.Figure:
  hourly = df["incident_hour"].dropna().value_counts().sort_index()
  fig = go.Figure(
      go.Bar(x=hourly.index, y=hourly.values, marker=dict(color=ACCENT_ORANGE))
  )
  fig.update_xaxes(dtick=2)
  return _style_fig(fig, CHART_H)


def fig_dow_hour_heatmap(df: pd.DataFrame) -> go.Figure:
  valid = df.dropna(subset=["incident_dow", "incident_hour"])
  heat = (
      valid.groupby(["incident_dow", "incident_hour"])
      .size()
      .unstack(fill_value=0)
  )
  heat = heat.reindex(DOW_ORDER)
  custom_scale = [[0.0, PANEL], [0.5, ACCENT_AMBER], [1.0, ACCENT_RED]]
  fig = go.Figure(
      go.Heatmap(
          z=heat.values,
          x=heat.columns,
          y=heat.index,
          colorscale=custom_scale,
          showscale=True,
          colorbar=dict(thickness=10, len=0.8, tickfont=dict(size=8)),
      )
  )
  return _style_fig(fig, CHART_H_LARGE)


def fig_geo_map(df: pd.DataFrame) -> go.Figure:
  geo = df.dropna(subset=["latitude", "longitude"])
  if len(geo) > 6000:
    geo = geo.sample(6000, random_state=42)

  density_scale = [
      [0.0, PANEL],
      [0.35, ACCENT_CYAN],
      [0.7, ACCENT_AMBER],
      [1.0, ACCENT_RED],
  ]
  common_kwargs = dict(
      data_frame=geo,
      lat="latitude",
      lon="longitude",
      radius=9,
      zoom=3,
      height=CHART_H_LARGE,
      color_continuous_scale=density_scale,
  )
  if hasattr(px, "density_map"):
    fig = px.density_map(**common_kwargs, map_style="carto-darkmatter")
  else:
    fig = px.density_mapbox(**common_kwargs, mapbox_style="carto-darkmatter")
  fig.update_layout(coloraxis_showscale=False)
  return _style_fig(fig, CHART_H_LARGE, show_legend=False)


def fig_age_histogram(df: pd.DataFrame, column: str, color: str) -> go.Figure:
  values = df[column].dropna()
  fig = go.Figure(
      go.Histogram(
          x=values,
          nbinsx=20,
          marker=dict(color=color, line=dict(color=BG, width=0.5)),
      )
  )
  return _style_fig(fig, CHART_H)


def fig_race_donut(df: pd.DataFrame) -> go.Figure:
  identified = df[df["suspect_id"].notna()].copy()
  identified["suspect_race"] = identified["suspect_race"].fillna("Unknown")
  counts = identified["suspect_race"].value_counts()
  fig = go.Figure(
      go.Pie(
          labels=counts.index,
          values=counts.values,
          hole=0.55,
          marker=dict(colors=QUALITATIVE_PALETTE),
          textinfo="percent",
          textfont=dict(size=10),
      )
  )
  return _style_fig(fig, CHART_H, show_legend=True)


def fig_property_loss_by_crime(df: pd.DataFrame, top_n: int = 8) -> go.Figure:
  top_crimes = df["crime_type"].value_counts().head(top_n).index
  loss = (
      df[df["crime_type"].isin(top_crimes)]
      .groupby("crime_type")["property_loss_usd"]
      .mean()
      .dropna()
      .sort_values(ascending=False)
  )
  fig = go.Figure(
      go.Bar(
          x=loss.index,
          y=loss.values,
          marker=dict(color=ACCENT_VIOLET),
      )
  )
  fig.update_yaxes(tickprefix="$")
  return _style_fig(fig, CHART_H)


def fig_reported_online_donut(df: pd.DataFrame) -> go.Figure:
  counts = df["reported_online"].astype(str).value_counts()
  color_map = {"True": ACCENT_GREEN, "False": ACCENT_RED, "Unknown": TEXT_MUTED}
  fig = go.Figure(
      go.Pie(
          labels=counts.index,
          values=counts.values,
          hole=0.55,
          marker=dict(
              colors=[color_map.get(k, ACCENT_CYAN) for k in counts.index]
          ),
          textinfo="percent",
          textfont=dict(size=10),
      )
  )
  return _style_fig(fig, CHART_H, show_legend=True)


def fig_num_arrests_bar(df: pd.DataFrame) -> go.Figure:
  counts = df["num_arrests"].dropna().astype(int).value_counts()
  full_range = range(0, 6)
  counts = counts.reindex(full_range, fill_value=0)
  fig = go.Figure(
      go.Bar(
          x=[str(i) for i in counts.index],
          y=counts.values,
          marker=dict(color=ACCENT_BLUE),
      )
  )
  fig.update_xaxes(type="category")
  return _style_fig(fig, CHART_H)


def fig_clusters_scatter(df: pd.DataFrame) -> go.Figure:
  sample = (
      df.dropna(subset=["longitude", "latitude", "crime_cluster"])
      .sample(min(len(df), 2500), random_state=42)
      .sort_values("crime_cluster")
  )
  fig = px.scatter(
      sample,
      x="longitude",
      y="latitude",
      color="crime_cluster",
      color_discrete_sequence=[
          ACCENT_CYAN,
          ACCENT_AMBER,
          ACCENT_VIOLET,
          ACCENT_GREEN,
      ],
      hover_data=["crime_type", "incident_hour", "property_loss_usd"],
  )
  fig.update_layout(
      title=None,
      margin=dict(l=40, r=28, t=28, b=8),
      legend=dict(
          orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=9)
      ),
  )
  return _style_fig(fig, CHART_H_LARGE, show_legend=True)


def fig_clusters_loss_hour(df: pd.DataFrame) -> go.Figure:
  cluster_stats = (
      df.groupby("crime_cluster")[["property_loss_usd", "incident_hour"]]
      .mean()
      .reset_index()
  )
  fig = go.Figure()
  fig.add_trace(
      go.Bar(
          name="Mean Loss ($)",
          x=cluster_stats["crime_cluster"],
          y=cluster_stats["property_loss_usd"],
          marker_color=ACCENT_AMBER,
      )
  )
  return _style_fig(fig, CHART_H_LARGE)


def render_overview_tab(df: pd.DataFrame) -> None:
  c1, c2, c3 = st.columns(3, gap="small")
  with c1:
    chart_card("Top Crime Types", fig_crime_type_bar(df))
  with c2:
    chart_card("Severity Breakdown", fig_severity_donut(df))
  with c3:
    chart_card("Incidents by District", fig_district_bar(df))

  c4, c5, c6 = st.columns(3, gap="small")
  with c4:
    chart_card("Case Status & Resolution", fig_case_status_resolution(df))
  with c5:
    chart_card("Weapon Used", fig_weapon_bar(df))
  with c6:
    chart_card("Suspect vs Victim Gender", fig_gender_comparison(df))


def render_time_geo_tab(df: pd.DataFrame) -> None:
  c1, c2, c3 = st.columns(3, gap="small")
  with c1:
    chart_card("Incidents per Year (Labeled)", fig_yearly_trend(df))
  with c2:
    chart_card("Incidents per Month", fig_monthly_bar(df))
  with c3:
    chart_card("Incidents by Hour of Day", fig_hourly_bar(df))

  c4, c5 = st.columns(2, gap="small")
  with c4:
    chart_card("Day of Week × Hour Heatmap", fig_dow_hour_heatmap(df))
  with c5:
    chart_card("Geographic Hotspots (Incident Density)", fig_geo_map(df))


def render_demographics_tab(df: pd.DataFrame) -> None:
  c1, c2, c3 = st.columns(3, gap="small")
  with c1:
    chart_card(
        "Suspect Age Distribution",
        fig_age_histogram(df, "suspect_age", ACCENT_RED),
    )
  with c2:
    chart_card(
        "Victim Age Distribution",
        fig_age_histogram(df, "victim_age", ACCENT_BLUE),
    )
  with c3:
    chart_card(
        "Suspect Race Breakdown (Identified Suspects)", fig_race_donut(df)
    )

  c4, c5, c6 = st.columns(3, gap="small")
  with c4:
    chart_card(
        "Avg Property Loss by Crime Type", fig_property_loss_by_crime(df)
    )
  with c5:
    chart_card("Reported Online", fig_reported_online_donut(df))
  with c6:
    chart_card("Number of Arrests", fig_num_arrests_bar(df))


def render_ml_clusters_tab(df: pd.DataFrame) -> None:
  st.markdown(
      "#### 🤖 Machine Learning Pattern Discovery (K-Means Clustering)"
  )
  st.caption(
      "Unsupervised clustering segmenting crimes into 4 distinct operational"
      " patterns based on Geography, Time, and Financial Damage."
  )

  c1, c2 = st.columns([1.5, 1], gap="small")
  with c1:
    chart_card("Cluster Spatial Breakdown", fig_clusters_scatter(df))
  with c2:
    chart_card("Cluster Average Loss ($)", fig_clusters_loss_hour(df))

  st.markdown("##### 📌 Cluster Profiles (Feature Averages)")
  summary = (
      df.groupby("crime_cluster")[
          ["property_loss_usd", "incident_hour", "latitude", "longitude"]
      ]
      .mean()
      .reset_index()
  )
  summary.columns = [
      "Cluster ID",
      "Mean Loss ($)",
      "Peak Hour",
      "Latitude",
      "Longitude",
  ]
  st.dataframe(
      summary.style.format({"Mean Loss ($)": "${:,.2f}"}),
      use_container_width=True,
  )

  st.markdown("---")
  st.markdown("#### 🎯 Interactive Crime Drilldown (ipywidgets Equivalent)")
  chosen_crime = st.selectbox(
      "Select Crime Type for Detailed Inspection:",
      options=sorted(df["crime_type"].dropna().unique()),
  )
  sub = df[df["crime_type"] == chosen_crime]

  w1, w2, w3, w4 = st.columns(4)
  w1.metric("Total Incidents", f"{len(sub):,}")
  w2.metric("Avg Property Loss", f"${sub['property_loss_usd'].mean():,.2f}")
  top_dist = (
      sub["district"].mode()[0] if not sub["district"].empty else "Unknown"
  )
  w3.metric("Top District", top_dist)
  top_weap = (
      sub["weapon_used"].mode()[0]
      if not sub["weapon_used"].empty
      else "Unknown"
  )
  w4.metric("Top Weapon", top_weap)


def render_header(filtered_count: int, total_count: int) -> None:
  st.markdown(
      f"""
        <div class="dash-header">
            <div>
                <p class="dash-title">🛡️ Crime Incident Command Center</p>
                <p class="dash-subtitle">Showing {filtered_count:,} of {total_count:,} incidents · filters apply across all tabs</p>
            </div>
        </div>
        """,
      unsafe_allow_html=True,
  )


def main() -> None:
  st.set_page_config(
      page_title="Crime Incident Command Center",
      page_icon="🛡️",
      layout="wide",
      initial_sidebar_state="expanded",
  )
  inject_custom_css()

  uploaded_file = None
  if not any(
      os.path.exists(p)
      for p in [DATA_FILENAME, DATA_FILENAME_FALLBACK, DATA_FILENAME_RAW]
  ):
    st.warning(
        f"`{DATA_FILENAME}` was not found next to crime_3.py. Upload it below"
        " to continue."
    )
    uploaded_file = st.file_uploader(
        "Upload crime_incidents_cleaned2.csv", type="csv"
    )
    if uploaded_file is None:
      st.stop()

  raw_df = load_raw_data(uploaded_file)
  if raw_df is None:
    st.stop()

  df = clean_crime_data(raw_df)

  filters = render_sidebar_filters(df)
  filtered_df = apply_filters(df, filters)

  render_header(len(filtered_df), len(df))

  if filtered_df.empty:
    st.info(
        "No incidents match the current filters. Try widening your selection or"
        " press **Reset Filters**."
    )
    st.stop()

  render_kpi_row(compute_kpis(filtered_df))

  tab1, tab2, tab3, tab4 = st.tabs([
      "📊 Overview",
      "🕒 Time & Geography",
      "👥 Demographics & Financial",
      "🤖 Machine Learning (Clusters)",
  ])
  with tab1:
    render_overview_tab(filtered_df)
  with tab2:
    render_time_geo_tab(filtered_df)
  with tab3:
    render_demographics_tab(filtered_df)
  with tab4:
    render_ml_clusters_tab(filtered_df)


if __name__ == "__main__":
  main()