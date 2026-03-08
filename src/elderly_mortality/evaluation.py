from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))


def stay_level_from_row(
    df_long: pd.DataFrame,
    row_score: np.ndarray,
    *,
    id_col: str = "stay_id",
    time_col: str = "t",
    label_col: str = "_future_label",
    cutoff_hours: int = 24,
    agg: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    if len(df_long) != len(row_score):
        raise ValueError(f"Length mismatch: df={len(df_long)} vs row_score={len(row_score)}")

    d = df_long.sort_values([id_col, time_col]).copy()
    d["_row_score"] = np.asarray(row_score, dtype=float)
    d = d.loc[d[time_col] <= cutoff_hours].copy()

    if agg == "max":
        s_stay = d.groupby(id_col)["_row_score"].max()
    elif agg == "mean":
        s_stay = d.groupby(id_col)["_row_score"].mean()
    elif agg == "last":
        s_stay = d.groupby(id_col)["_row_score"].last()
    else:
        raise ValueError("agg must be one of {'max', 'mean', 'last'}")

    y_stay = d.groupby(id_col)[label_col].max().astype(int)
    common = s_stay.index.intersection(y_stay.index)
    return y_stay.loc[common].to_numpy(), s_stay.loc[common].to_numpy()


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= float(thr)).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def pick_threshold_by_recall_then_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_recall: float = 0.80,
    n_grid: int = 1001,
) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    lo, hi = float(np.nanmin(y_score)), float(np.nanmax(y_score))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        thr = lo if np.isfinite(lo) else 0.5
        m = metrics_at_threshold(y_true, y_score, thr)
        return float(thr), float(m["precision"]), float(m["recall"])

    thrs = np.linspace(lo, hi, n_grid)
    best: tuple[tuple[float, float], float, float, float] | None = None

    for thr in thrs:
        m = metrics_at_threshold(y_true, y_score, float(thr))
        rec = float(m["recall"])
        prec = float(m["precision"])
        if rec + 1e-12 >= float(target_recall):
            key = (prec, -float(thr))
            if best is None or key > best[0]:
                best = (key, float(thr), prec, rec)

    if best is not None:
        _, thr, prec, rec = best
        return float(thr), float(prec), float(rec)

    fallback: tuple[tuple[float, float], float, float, float] | None = None
    for thr in thrs:
        m = metrics_at_threshold(y_true, y_score, float(thr))
        rec = float(m["recall"])
        prec = float(m["precision"])
        key = (rec, prec)
        if fallback is None or key > fallback[0]:
            fallback = (key, float(thr), prec, rec)

    assert fallback is not None
    _, thr, prec, rec = fallback
    return float(thr), float(prec), float(rec)

