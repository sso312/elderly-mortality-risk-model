import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="ICU Risk Monitor", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("STREAMLIT_DATA_DIR", str(BASE_DIR / "data")))
HEADER_PATH = DATA_DIR / "sample_patient_header.csv"
TS_PATH = DATA_DIR / "sample_timeseries.csv"
RISK_THRESHOLD = 0.30


def _normalize_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "stayid" not in out.columns and "stay_id" in out.columns:
        out = out.rename(columns={"stay_id": "stayid"})

    if "t" not in out.columns:
        raise KeyError("Timeseries must contain 't' column.")
    if "stayid" not in out.columns:
        raise KeyError("Timeseries must contain 'stayid' or 'stay_id' column.")

    out["stayid"] = pd.to_numeric(out["stayid"], errors="coerce").astype("Int64")
    out["t"] = pd.to_numeric(out["t"], errors="coerce")

    if "risk_score" not in out.columns:
        if "risk_score_6h" in out.columns:
            out["risk_score"] = pd.to_numeric(out["risk_score_6h"], errors="coerce").fillna(0.0)
        elif "_future_label" in out.columns:
            out["risk_score"] = pd.to_numeric(out["_future_label"], errors="coerce").fillna(0.0)
        else:
            out["risk_score"] = 0.0

    # Compatibility column aliases.
    if "RespRatestd6h" not in out.columns and "RespRate_std_6h" in out.columns:
        out["RespRatestd6h"] = pd.to_numeric(out["RespRate_std_6h"], errors="coerce")
    if "HeartRatestd6h" not in out.columns and "HeartRate_std_6h" in out.columns:
        out["HeartRatestd6h"] = pd.to_numeric(out["HeartRate_std_6h"], errors="coerce")
    if "Tempstd6h" not in out.columns and "Temp_std_6h" in out.columns:
        out["Tempstd6h"] = pd.to_numeric(out["Temp_std_6h"], errors="coerce")

    # FiO2 ratio -> percentage if needed.
    if "FiO2" in out.columns:
        fio2 = pd.to_numeric(out["FiO2"], errors="coerce")
        mx = fio2.max()
        if pd.notna(mx) and mx <= 1.0:
            out["FiO2"] = fio2 * 100.0
        else:
            out["FiO2"] = fio2

    return out.sort_values(["stayid", "t"]).reset_index(drop=True)


@st.cache_data
def load_header() -> pd.DataFrame:
    if not HEADER_PATH.exists():
        raise FileNotFoundError(f"Missing header file: {HEADER_PATH}")
    h = pd.read_csv(HEADER_PATH)
    if "stayid" not in h.columns and "stay_id" in h.columns:
        h = h.rename(columns={"stay_id": "stayid"})
    if "stayid" not in h.columns:
        raise KeyError("Header CSV must include 'stayid' or 'stay_id'.")
    h["stayid"] = pd.to_numeric(h["stayid"], errors="coerce").astype("Int64")
    if "name" not in h.columns:
        h["name"] = h["stayid"].apply(lambda v: f"PATIENT_{v}")
    return h


@st.cache_data
def load_timeseries() -> pd.DataFrame:
    if not TS_PATH.exists():
        raise FileNotFoundError(f"Missing timeseries file: {TS_PATH}")
    return _normalize_timeseries(pd.read_csv(TS_PATH))


def latest_risk_by_stay(ts: pd.DataFrame) -> dict[int, float]:
    latest = ts.sort_values("t").groupby("stayid", as_index=False).tail(1)
    out: dict[int, float] = {}
    for _, row in latest.iterrows():
        sid = int(row["stayid"])
        out[sid] = float(pd.to_numeric(row.get("risk_score", 0.0), errors="coerce") or 0.0)
    return out


def draw_line(df: pd.DataFrame, y_col: str, title: str, color: str) -> None:
    if y_col not in df.columns:
        st.info(f"Column not found: {y_col}")
        return
    d = df[["t", y_col]].dropna()
    if d.empty:
        st.info(f"No data: {y_col}")
        return
    chart = (
        alt.Chart(d)
        .mark_line(color=color)
        .encode(x=alt.X("t:Q", title="Hours"), y=alt.Y(f"{y_col}:Q", title=title, scale=alt.Scale(zero=False)))
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)


header_df = load_header()
ts_df = load_timeseries()
risk_map = latest_risk_by_stay(ts_df)

st.title("ICU Patient Risk Dashboard")
st.caption(f"Data directory: {DATA_DIR}")

with st.sidebar:
    st.subheader("Patients")
    q = st.text_input("Search")
    all_ids = [int(v) for v in header_df["stayid"].dropna().astype(int).unique()]
    n_critical = sum(1 for sid in all_ids if risk_map.get(sid, 0.0) >= RISK_THRESHOLD)
    n_stable = len(all_ids) - n_critical
    flt = st.radio("Status", [f"All ({len(all_ids)})", f"Critical ({n_critical})", f"Stable ({n_stable})"])

    pick_df = header_df.copy()
    if "Critical" in flt:
        pick_ids = [sid for sid in all_ids if risk_map.get(sid, 0.0) >= RISK_THRESHOLD]
        pick_df = pick_df[pick_df["stayid"].astype(int).isin(pick_ids)]
    elif "Stable" in flt:
        pick_ids = [sid for sid in all_ids if risk_map.get(sid, 0.0) < RISK_THRESHOLD]
        pick_df = pick_df[pick_df["stayid"].astype(int).isin(pick_ids)]

    if q:
        key = pick_df["name"].astype(str) + " " + pick_df["stayid"].astype(str)
        pick_df = pick_df[key.str.contains(q, case=False, na=False)]

    if pick_df.empty:
        st.warning("No patient matches current filter.")
        st.stop()

    options = [int(v) for v in pick_df["stayid"].dropna().astype(int).tolist()]

    def fmt(sid: int) -> str:
        row = header_df[header_df["stayid"].astype(int) == int(sid)].iloc[0]
        risk = risk_map.get(int(sid), 0.0)
        marker = "CRIT" if risk >= RISK_THRESHOLD else "OK"
        return f"[{marker}] {row['name']} ({sid})"

    selected_stayid = st.radio("Select", options, format_func=fmt)

pt_h = header_df[header_df["stayid"].astype(int) == int(selected_stayid)].iloc[0]
pt_ts = ts_df[ts_df["stayid"].astype(int) == int(selected_stayid)].sort_values("t")
if pt_ts.empty:
    st.error("No timeseries rows for selected patient.")
    st.stop()
latest = pt_ts.iloc[-1]
risk = float(risk_map.get(int(selected_stayid), 0.0))

badge = "Critical" if risk >= RISK_THRESHOLD else "Stable"
st.subheader(f"{pt_h.get('name', 'Unknown')} ({int(selected_stayid)})")
st.write(
    f"Sex/Age: {pt_h.get('sex', 'M')}/{pt_h.get('age', '-')}, "
    f"Weight: {pt_h.get('weight', '-')}, Ward: {pt_h.get('ward', 'ICU')}, "
    f"Risk: {risk*100:.1f}% ({badge})"
)

mode = st.radio("View", ["Clinician", "Patient/Family"], horizontal=True)

if mode == "Clinician":
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### Vital Trends (last 24h)")
        tmax = pd.to_numeric(pt_ts["t"], errors="coerce").max()
        plot_df = pt_ts[pt_ts["t"] >= (tmax - 24)].copy()
        tab1, tab2, tab3 = st.tabs(["Circulatory", "Respiratory", "Neurologic"])
        with tab1:
            draw_line(plot_df, "SysBP", "SysBP", "#d9480f")
            draw_line(plot_df, "MeanBP", "MeanBP", "#e8590c")
        with tab2:
            draw_line(plot_df, "FiO2", "FiO2 (%)", "#1971c2")
        with tab3:
            draw_line(plot_df, "GCS_Total", "GCS_Total", "#f08c00")

        st.markdown("### Horizon Risk")
        k1, k2, k3 = st.columns(3)
        v6 = latest.get("risk_score_6h", latest.get("risk_score", np.nan))
        v12 = latest.get("risk_score_12h", np.nan)
        v24 = latest.get("risk_score_24h", np.nan)
        k1.metric("6h", f"{float(v6)*100:.1f}%" if pd.notna(v6) else "-")
        k2.metric("12h", f"{float(v12)*100:.1f}%" if pd.notna(v12) else "-")
        k3.metric("24h", f"{float(v24)*100:.1f}%" if pd.notna(v24) else "-")

    with c2:
        st.markdown("### Last 6h Variability")
        std_cols = [c for c in ["RespRatestd6h", "HeartRatestd6h", "Tempstd6h"] if c in pt_ts.columns]
        if not std_cols:
            st.info("No std6h columns available.")
        else:
            rows = []
            for c in std_cols:
                rows.append({"metric": c, "value": float(pd.to_numeric(latest.get(c, np.nan), errors="coerce"))})
            std_df = pd.DataFrame(rows)
            st.dataframe(std_df, use_container_width=True, hide_index=True)

        if "risk_score" in pt_ts.columns:
            d = pt_ts[["t", "risk_score"]].dropna()
            if not d.empty:
                risk_chart = (
                    alt.Chart(d)
                    .mark_area(opacity=0.3, color="#fa5252")
                    .encode(x=alt.X("t:Q", title="Hours"), y=alt.Y("risk_score:Q", title="Risk", scale=alt.Scale(domain=[0, 1])))
                    .properties(height=160)
                )
                st.altair_chart(risk_chart, use_container_width=True)
else:
    st.info("Simplified patient/family view.")
    st.metric("Current Risk", f"{risk*100:.1f}%")
    st.write("Current major values")
    m1, m2, m3 = st.columns(3)
    m1.metric("SysBP", f"{float(latest.get('SysBP', np.nan)):.1f}" if pd.notna(latest.get("SysBP", np.nan)) else "-")
    m2.metric("MeanBP", f"{float(latest.get('MeanBP', np.nan)):.1f}" if pd.notna(latest.get("MeanBP", np.nan)) else "-")
    m3.metric("FiO2", f"{float(latest.get('FiO2', np.nan)):.1f}%" if pd.notna(latest.get("FiO2", np.nan)) else "-")

