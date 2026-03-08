from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TREND_VARIABLES


def add_gcs_total(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = ["GCS_Eye", "GCS_Verbal", "GCS_Motor"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        return out

    for c in required:
        m = f"{c}_measured"
        if m not in out.columns:
            out[m] = out[c].notna().astype(int)

    out["GCS_Total_measured"] = (
        (out["GCS_Eye_measured"] == 1)
        & (out["GCS_Verbal_measured"] == 1)
        & (out["GCS_Motor_measured"] == 1)
    ).astype(int)
    out["GCS_Total"] = np.where(
        out["GCS_Total_measured"] == 1,
        out["GCS_Eye"] + out["GCS_Verbal"] + out["GCS_Motor"],
        np.nan,
    )
    return out


def add_trend_features(
    df: pd.DataFrame,
    *,
    variables: list[str] | None = None,
    id_col: str = "stay_id",
    time_col: str = "t",
    window: int = 6,
) -> pd.DataFrame:
    out = df.sort_values([id_col, time_col]).copy()
    variables = variables or list(TREND_VARIABLES)

    for var in variables:
        if var not in out.columns:
            continue
        grp = out.groupby(id_col, sort=False)[var]
        out[f"{var}_diff"] = grp.diff()
        out[f"{var}_mean_6h"] = (
            grp.rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        out[f"{var}_std_6h"] = (
            grp.rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True)
        )

    trend_cols = [
        c for c in out.columns if c.endswith("_diff") or c.endswith("_mean_6h") or c.endswith("_std_6h")
    ]
    if trend_cols:
        out[trend_cols] = out[trend_cols].fillna(0)
    return out


def build_feature_columns(
    df: pd.DataFrame,
    *,
    id_cols: list[str] | None = None,
    label_cols: list[str] | None = None,
    aux_cols: list[str] | None = None,
    weight_cols: list[str] | None = None,
    extra_drop: set[str] | None = None,
) -> list[str]:
    id_cols = id_cols or ["subject_id", "stay_id", "t"]
    label_cols = label_cols or ["event", "delta"]
    aux_cols = aux_cols or ["icu_mortality"]
    weight_cols = weight_cols or ["sample_weight"]
    extra_drop = extra_drop or set()
    drop_cols = set(id_cols + label_cols + aux_cols + weight_cols) | set(extra_drop)
    return [c for c in df.columns if c not in drop_cols]

