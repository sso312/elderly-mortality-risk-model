from __future__ import annotations

import numpy as np
import pandas as pd


def add_future_label(
    df: pd.DataFrame,
    *,
    id_col: str = "stay_id",
    time_col: str = "t",
    event_col: str = "event",
    horizon_hours: int = 6,
    label_col: str = "_future_label",
) -> pd.DataFrame:
    out = df.copy()
    out[label_col] = 0

    for _, group in out.sort_values([id_col, time_col]).groupby(id_col, sort=False):
        t = group[time_col].to_numpy()
        e = group[event_col].to_numpy().astype(int)
        event_times = t[e == 1]
        if event_times.size == 0:
            continue
        left = np.searchsorted(event_times, t, side="right")
        right = np.searchsorted(event_times, t + horizon_hours, side="right")
        out.loc[group.index, label_col] = ((right - left) > 0).astype(int)
    return out


def add_label_observable_mask(
    df: pd.DataFrame,
    *,
    id_col: str = "stay_id",
    time_col: str = "t",
    label_col: str = "_future_label",
    horizon_hours: int = 6,
    mask_col: str = "_label_observable",
) -> pd.DataFrame:
    out = df.copy()
    t_last = out.groupby(id_col)[time_col].max().rename("_t_last")
    out = out.merge(t_last, on=id_col, how="left")
    out[mask_col] = (out[label_col] == 1) | (out[time_col] + horizon_hours <= out["_t_last"])
    out = out.drop(columns=["_t_last"])
    return out


def drop_rows_after_first_event(
    df: pd.DataFrame,
    *,
    id_col: str = "stay_id",
    time_col: str = "t",
    event_col: str = "event",
) -> pd.DataFrame:
    out = df.copy()
    first_t = out.loc[out[event_col].astype(int) == 1].groupby(id_col)[time_col].min()
    out = out.merge(first_t.rename("_first_event_t"), on=id_col, how="left")
    keep = out["_first_event_t"].isna() | (out[time_col] < out["_first_event_t"])
    return out.loc[keep].drop(columns=["_first_event_t"])


def apply_horizon_label_pipeline(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    horizon_hours: int = 6,
    drop_after_event: bool = True,
    event_col: str = "event",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _one(df: pd.DataFrame) -> pd.DataFrame:
        out = add_future_label(df, event_col=event_col, horizon_hours=horizon_hours)
        out = add_label_observable_mask(out, horizon_hours=horizon_hours)
        out = out.loc[out["_label_observable"]].copy()
        if drop_after_event:
            out = drop_rows_after_first_event(out, event_col=event_col)
        return out

    return _one(train_df), _one(valid_df), _one(test_df)

