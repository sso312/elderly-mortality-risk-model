from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def _standardize_stay_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "stayid" not in out.columns and "stay_id" in out.columns:
        out = out.rename(columns={"stay_id": "stayid"})
    return out


def _to_risk_score(df: pd.DataFrame) -> pd.Series:
    if "risk_score" in df.columns:
        return pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0)
    if "risk_score_6h" in df.columns:
        return pd.to_numeric(df["risk_score_6h"], errors="coerce").fillna(0.0)
    if "_future_label" in df.columns:
        return pd.to_numeric(df["_future_label"], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(df), dtype=float), index=df.index)


def build_streamlit_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    out = _standardize_stay_col(df)
    if "stayid" not in out.columns:
        raise KeyError("Source dataframe must contain 'stay_id' or 'stayid'.")
    if "t" not in out.columns:
        raise KeyError("Source dataframe must contain 't'.")

    rename_map = {
        "RespRate_std_6h": "RespRatestd6h",
        "HeartRate_std_6h": "HeartRatestd6h",
        "Temp_std_6h": "Tempstd6h",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    out["risk_score"] = _to_risk_score(out)

    # If FiO2 is in ratio scale [0, 1], convert to percentage.
    if "FiO2" in out.columns:
        fio2 = pd.to_numeric(out["FiO2"], errors="coerce")
        mx = fio2.max()
        if pd.notna(mx) and mx <= 1.0:
            out["FiO2"] = fio2 * 100.0
        else:
            out["FiO2"] = fio2

    # App compatibility defaults.
    if "DiasBP" not in out.columns and "DiasBP_mean_6h" in out.columns:
        out["DiasBP"] = pd.to_numeric(out["DiasBP_mean_6h"], errors="coerce")
    if "GCS_Total" not in out.columns and "GCS_Total_mean_6h" in out.columns:
        out["GCS_Total"] = pd.to_numeric(out["GCS_Total_mean_6h"], errors="coerce")

    preferred_cols = [
        "stayid",
        "t",
        "risk_score",
        "risk_score_6h",
        "risk_score_12h",
        "risk_score_24h",
        "SysBP",
        "MeanBP",
        "DiasBP",
        "FiO2",
        "GCS_Total",
        "pH",
        "RespRatestd6h",
        "HeartRatestd6h",
        "Tempstd6h",
        "_future_label",
    ]
    present = [c for c in preferred_cols if c in out.columns]
    if not present:
        raise RuntimeError("No compatible columns found for streamlit timeseries export.")
    return out[present].sort_values(["stayid", "t"]).reset_index(drop=True)


def build_streamlit_header(df_ts: pd.DataFrame) -> pd.DataFrame:
    d = _standardize_stay_col(df_ts)
    if "stayid" not in d.columns:
        raise KeyError("Timeseries dataframe must contain 'stayid'.")

    rows = []
    today = date.today().isoformat()
    for sid, g in d.groupby("stayid", sort=True):
        g_last = g.sort_values("t").iloc[-1]
        age = int(g_last["age"]) if "age" in g_last and pd.notna(g_last["age"]) else 86
        weight = float(g_last["Weight"]) if "Weight" in g_last and pd.notna(g_last["Weight"]) else 75.0
        sex = str(g_last["sex"]) if "sex" in g_last and pd.notna(g_last["sex"]) else "M"
        rows.append(
            {
                "stayid": int(sid),
                "name": f"PATIENT_{int(sid)}",
                "sex": sex,
                "age": age,
                "weight": round(weight, 1),
                "admit_date": today,
                "ward": "ICU",
                "staff_code": "DR_177",
            }
        )
    return pd.DataFrame(rows)


def export_streamlit_bundle(
    source_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    n_patients: int = 20,
    seed: int = 42,
) -> tuple[Path, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ts = build_streamlit_timeseries(source_df)
    uniq = ts["stayid"].dropna().astype(int).unique()
    if len(uniq) > n_patients:
        rng = np.random.default_rng(seed)
        pick = set(int(v) for v in rng.choice(uniq, size=n_patients, replace=False))
        ts = ts[ts["stayid"].astype(int).isin(pick)].copy()

    header = build_streamlit_header(ts)
    ts_file = out_path / "sample_timeseries.csv"
    hd_file = out_path / "sample_patient_header.csv"
    ts.to_csv(ts_file, index=False, encoding="utf-8-sig")
    header.to_csv(hd_file, index=False, encoding="utf-8-sig")
    return hd_file, ts_file

