from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elderly_mortality.evaluation import metrics_at_threshold, pick_threshold_by_recall_then_precision
from elderly_mortality.features import build_feature_columns
from elderly_mortality.models.boxhed_wrapper import (
    aggregate_stay_scores_from_boxhed_df,
    make_boxhed,
    prepare_for_boxhed,
    safe_cv,
    safe_fit,
    safe_predict_scores,
    safe_preprocess,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BoXHED experiment using pre-split CSV files.")
    p.add_argument("--input-dir", required=True, help="Folder containing train_df.csv, valid_df.csv, test_df.csv")
    p.add_argument("--out-dir", required=True, help="Folder for metrics/predictions")
    p.add_argument("--target-recall", type=float, default=0.80)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--score-agg",
        choices=["max", "mean"],
        default="max",
        help="How to aggregate row scores to stay-level scores.",
    )
    return p.parse_args()


def _safe_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(input_dir / "train_df.csv")
    valid_df = pd.read_csv(input_dir / "valid_df.csv")
    test_df = pd.read_csv(input_dir / "test_df.csv")

    feature_cols = build_feature_columns(
        train_df,
        label_cols=["event", "delta"],
        aux_cols=["icu_mortality"],
        weight_cols=["sample_weight"],
    )

    train_bx = prepare_for_boxhed(train_df, feature_cols)
    valid_bx = prepare_for_boxhed(valid_df, feature_cols)
    test_bx = prepare_for_boxhed(test_df, feature_cols)

    bx = make_boxhed()
    train_data = safe_preprocess(bx, train_bx, weighted=False)
    valid_data = safe_preprocess(bx, valid_bx, weighted=False)
    test_data = safe_preprocess(bx, test_bx, weighted=False)

    param_grid = {
        "n_estimators": [400, 800, 1200],
        "max_depth": [2, 3, 4],
        "eta": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "lambda": [0.0, 1.0, 5.0],
    }

    cv_res = safe_cv(
        train_data=train_data,
        param_grid=param_grid,
        num_folds=args.num_folds,
        gpu_list=[0],
        seed=args.seed,
    )

    best_params = cv_res
    if isinstance(cv_res, dict) and "best_param_1se" in cv_res:
        best_params = cv_res["best_param_1se"]
    else:
        try:
            from boxhed.model_selection import best_param_1se_rule

            best_params = best_param_1se_rule(cv_res)
        except Exception:
            best_params = param_grid

    bx_final = make_boxhed()
    _ = safe_fit(bx_final, train_data, best_params, valid_data=valid_data)

    valid_row_score = safe_predict_scores(bx_final, valid_data)
    test_row_score = safe_predict_scores(bx_final, test_data)

    yv, sv = aggregate_stay_scores_from_boxhed_df(valid_bx, valid_row_score, score_agg=args.score_agg)
    yt, st = aggregate_stay_scores_from_boxhed_df(test_bx, test_row_score, score_agg=args.score_agg)

    thr, _, _ = pick_threshold_by_recall_then_precision(yv, sv, target_recall=args.target_recall)
    valid_auc, valid_ap = _safe_auc_ap(yv, sv)
    test_auc, test_ap = _safe_auc_ap(yt, st)
    valid_m = metrics_at_threshold(yv, sv, thr)
    test_m = metrics_at_threshold(yt, st, thr)

    summary = {
        "target_recall": args.target_recall,
        "score_agg": args.score_agg,
        "threshold": thr,
        "valid_auc": valid_auc,
        "valid_ap": valid_ap,
        "valid_precision": valid_m["precision"],
        "valid_recall": valid_m["recall"],
        "valid_f1": valid_m["f1"],
        "test_auc": test_auc,
        "test_ap": test_ap,
        "test_precision": test_m["precision"],
        "test_recall": test_m["recall"],
        "test_f1": test_m["f1"],
        "n_valid_stay": int(len(yv)),
        "n_test_stay": int(len(yt)),
    }

    (out_dir / "boxhed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame({"y_true": yv, "score": sv}).to_csv(out_dir / "valid_stay_scores.csv", index=False)
    pd.DataFrame({"y_true": yt, "score": st}).to_csv(out_dir / "test_stay_scores.csv", index=False)

    print("[DONE] BoXHED experiment complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
