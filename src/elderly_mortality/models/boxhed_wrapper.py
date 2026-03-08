from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd

try:  # pragma: no cover
    from boxhed.boxhed import boxhed
    from boxhed.model_selection import best_param_1se_rule, cv
except Exception:  # pragma: no cover
    boxhed = None
    best_param_1se_rule = None
    cv = None


def require_boxhed() -> None:
    if boxhed is None or cv is None or best_param_1se_rule is None:
        raise ImportError(
            "BoXHED package is not available. Install and verify imports for "
            "`from boxhed.boxhed import boxhed` and `from boxhed.model_selection import cv`."
        )


def make_boxhed() -> Any:
    require_boxhed()
    return boxhed()


def prepare_for_boxhed(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    exclude_features: list[str] | None = None,
    id_col: str = "stay_id",
    time_col: str = "t",
    delta_col: str = "delta",
    event_col: str = "event",
    weight_col: str = "sample_weight",
    dt: int = 1,
) -> pd.DataFrame:
    required = {id_col, time_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()
    if delta_col not in out.columns:
        if event_col not in out.columns:
            raise KeyError(f"Neither '{delta_col}' nor '{event_col}' exists.")
        out[delta_col] = out[event_col]

    final_features = list(feature_cols)
    if exclude_features:
        ex = set(exclude_features)
        final_features = [f for f in final_features if f not in ex]
    if not final_features:
        raise ValueError("No features left after exclusion.")

    missing_feat = sorted(set(final_features) - set(out.columns))
    if missing_feat:
        raise KeyError(f"Missing feature columns in input df: {missing_feat[:20]}")

    out["ID"] = out[id_col].astype(int)
    out["t_start"] = out[time_col].astype(int)
    out["t_end"] = (out["t_start"] + int(dt)).astype(int)
    out[delta_col] = out[delta_col].astype(int)

    keep_cols = ["ID", "t_start", "t_end", delta_col] + final_features
    if weight_col in out.columns:
        keep_cols.append(weight_col)
        out[weight_col] = out[weight_col].astype(float)

    return out[keep_cols].copy()


def safe_preprocess(
    bx_obj: Any,
    data: pd.DataFrame,
    *,
    is_cat: list[int] | None = None,
    num_quantiles: int = 256,
    weighted: bool = False,
    nthread: int = -1,
) -> Any:
    is_cat = is_cat or []
    sig = inspect.signature(bx_obj.preprocess)
    params = sig.parameters
    kwargs: dict[str, Any] = {}

    candidates = [
        ("data", data),
        ("df", data),
        ("is_cat", is_cat),
        ("num_quantiles", num_quantiles),
        ("weighted", weighted),
        ("nthread", nthread),
        ("nthreads", nthread),
        ("n_jobs", nthread),
    ]
    for k, v in candidates:
        if k in params:
            kwargs[k] = v

    if not any(k in kwargs for k in ("data", "df")):
        return bx_obj.preprocess(data, **kwargs)
    return bx_obj.preprocess(**kwargs)


def safe_cv(
    train_data: Any,
    param_grid: dict[str, Any],
    *,
    num_folds: int = 5,
    gpu_list: list[int] | None = None,
    batch_size: int = 128,
    seed: int = 42,
    nthread: int = -1,
    verbose: int = 1,
    early_stopping_rounds: int = 50,
    num_boost_round: int = 2000,
) -> Any:
    require_boxhed()
    if "n_estimators" not in param_grid:
        raise KeyError("param_grid must include 'n_estimators'.")

    gpu_list = gpu_list or [0]
    if not gpu_list:
        gpu_list = [0]
    batch_size = max(int(batch_size), len(gpu_list))

    sig = inspect.signature(cv)
    params = sig.parameters
    extra = {}
    for k, v in {
        "seed": seed,
        "nthread": nthread,
        "verbose": verbose,
        "early_stopping_rounds": early_stopping_rounds,
        "num_boost_round": num_boost_round,
        "batch_size": batch_size,
    }.items():
        if k in params:
            extra[k] = v

    call_kwargs: dict[str, Any] = {}
    if "param_grid" in params:
        call_kwargs["param_grid"] = param_grid
    if "data" in params:
        call_kwargs["data"] = train_data
    if "num_folds" in params:
        call_kwargs["num_folds"] = int(num_folds)
    if "gpu_list" in params:
        call_kwargs["gpu_list"] = list(gpu_list)
    call_kwargs.update(extra)

    try:
        if all(k in call_kwargs for k in ["param_grid", "data", "num_folds", "gpu_list"]):
            return cv(**call_kwargs)
    except TypeError:
        pass
    try:
        return cv(param_grid, train_data, int(num_folds), list(gpu_list), **extra)
    except TypeError:
        return cv(train_data, param_grid, int(num_folds), list(gpu_list), **extra)


def safe_fit(bx_obj: Any, train_data: Any, params: dict[str, Any], valid_data: Any | None = None) -> Any:
    sig = inspect.signature(bx_obj.fit)
    p = sig.parameters
    kwargs: dict[str, Any] = {}

    if "data" in p:
        kwargs["data"] = train_data
    if "params" in p:
        kwargs["params"] = params
    elif "param" in p:
        kwargs["param"] = params

    if valid_data is not None:
        for k in ["valid_data", "eval_data", "val_data", "data_valid"]:
            if k in p:
                kwargs[k] = valid_data
                break

    if "data" in kwargs and ("params" in kwargs or "param" in kwargs):
        return bx_obj.fit(**kwargs)
    if valid_data is not None:
        return bx_obj.fit(train_data, params, valid_data)
    return bx_obj.fit(train_data, params)


def safe_predict_scores(bx_obj: Any, data: Any) -> np.ndarray:
    for fn in ["predict_proba", "predict_score", "predict"]:
        if hasattr(bx_obj, fn):
            out = np.asarray(getattr(bx_obj, fn)(data))
            if out.ndim == 2 and out.shape[1] >= 2:
                return out[:, 1].astype(float)
            return out.reshape(-1).astype(float)
    raise AttributeError("No usable predict method found on BoXHED object.")


def aggregate_stay_scores_from_boxhed_df(
    df_bx: pd.DataFrame,
    row_scores: np.ndarray,
    *,
    id_col: str = "ID",
    delta_col: str = "delta",
    score_agg: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    tmp = df_bx[[id_col, delta_col]].copy()
    tmp["score"] = np.asarray(row_scores, dtype=float)

    if score_agg == "max":
        s_stay = tmp.groupby(id_col)["score"].max()
    elif score_agg == "mean":
        s_stay = tmp.groupby(id_col)["score"].mean()
    else:
        raise ValueError("score_agg must be one of {'max', 'mean'}")

    y_stay = tmp.groupby(id_col)[delta_col].max().astype(int).loc[s_stay.index]
    return y_stay.to_numpy(), s_stay.to_numpy()

