from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elderly_mortality.config import DEFAULT_EXTRA_DROP
from elderly_mortality.evaluation import (
    metrics_at_threshold,
    pick_threshold_by_recall_then_precision,
    safe_auc_ap,
    stay_level_from_row,
)
from elderly_mortality.features import build_feature_columns
from elderly_mortality.labels import apply_horizon_label_pipeline
from elderly_mortality.models.lstm_timewise import predict_row_score_lstm, train_lstm_row_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LSTM experiment using pre-split CSV files.")
    p.add_argument("--input-dir", required=True, help="Folder containing train_df.csv, valid_df.csv, test_df.csv")
    p.add_argument("--out-dir", required=True, help="Folder for metrics/predictions")
    p.add_argument("--horizon-hours", type=int, default=6)
    p.add_argument("--cutoff-hours", type=int, default=24)
    p.add_argument("--target-recall", type=float, default=0.80)
    p.add_argument("--max-len", type=int, default=120)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--event-col", default="event")
    p.add_argument("--drop-after-event", action="store_true", default=True)
    p.add_argument("--no-drop-after-event", dest="drop_after_event", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(input_dir / "train_df.csv")
    valid_raw = pd.read_csv(input_dir / "valid_df.csv")
    test_raw = pd.read_csv(input_dir / "test_df.csv")

    train_df, valid_df, test_df = apply_horizon_label_pipeline(
        train_raw,
        valid_raw,
        test_raw,
        horizon_hours=args.horizon_hours,
        drop_after_event=args.drop_after_event,
        event_col=args.event_col,
    )

    feature_cols = build_feature_columns(
        train_df,
        label_cols=["event", "delta", "_future_label"],
        aux_cols=["icu_mortality"],
        extra_drop=DEFAULT_EXTRA_DROP,
    )

    model, scaler = train_lstm_row_model(
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
        label_col="_future_label",
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )

    valid_score = predict_row_score_lstm(
        model, scaler, valid_df, feature_cols, label_col="_future_label", max_len=args.max_len
    )
    test_score = predict_row_score_lstm(
        model, scaler, test_df, feature_cols, label_col="_future_label", max_len=args.max_len
    )

    yv, sv = stay_level_from_row(
        valid_df,
        valid_score,
        label_col="_future_label",
        cutoff_hours=args.cutoff_hours,
        agg="max",
    )
    yt, st = stay_level_from_row(
        test_df,
        test_score,
        label_col="_future_label",
        cutoff_hours=args.cutoff_hours,
        agg="max",
    )

    thr, _, _ = pick_threshold_by_recall_then_precision(yv, sv, target_recall=args.target_recall)
    valid_auc, valid_ap = safe_auc_ap(yv, sv)
    test_auc, test_ap = safe_auc_ap(yt, st)
    valid_m = metrics_at_threshold(yv, sv, thr)
    test_m = metrics_at_threshold(yt, st, thr)

    summary = {
        "horizon_hours": args.horizon_hours,
        "cutoff_hours": args.cutoff_hours,
        "target_recall": args.target_recall,
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

    (out_dir / "lstm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame({"y_true": yv, "score": sv}).to_csv(out_dir / "valid_stay_scores.csv", index=False)
    pd.DataFrame({"y_true": yt, "score": st}).to_csv(out_dir / "test_stay_scores.csv", index=False)

    print("[DONE] LSTM experiment complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
