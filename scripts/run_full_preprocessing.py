from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elderly_mortality.config import DEFAULT_EXTRA_DROP, PreprocessConfig
from elderly_mortality.features import add_gcs_total, add_trend_features, build_feature_columns
from elderly_mortality.mimic_duckdb import build_full_model_table, register_raw_views, summarize_label_balance
from elderly_mortality.split import add_sample_weight_by_event, split_by_subject_stratified


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build full preprocessing outputs from raw MIMIC CSV files.")
    p.add_argument("--data-root", required=True, help="Root folder containing MIMIC CSV files.")
    p.add_argument("--out-dir", required=True, help="Output folder for split CSVs.")
    p.add_argument("--max-hours", type=int, default=120)
    p.add_argument("--min-age", type=int, default=65)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--weight-mode",
        default="ratio",
        choices=["ratio", "sqrt_ratio"],
        help="Class weighting mode for sample_weight.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PreprocessConfig(max_hours=args.max_hours, min_age=args.min_age, random_state=args.random_state)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    selected = register_raw_views(con, args.data_root)

    model_df, _ = build_full_model_table(con, max_hours=cfg.max_hours, min_age=cfg.min_age)
    model_df = add_gcs_total(model_df)
    model_df = add_trend_features(model_df, variables=list(cfg.trend_variables))

    split = split_by_subject_stratified(
        model_df,
        random_state=cfg.random_state,
        test_size=cfg.test_size,
        valid_size_within_train=cfg.valid_size / (cfg.train_size + cfg.valid_size),
    )
    train_df, valid_df, test_df, pos_weight = add_sample_weight_by_event(
        split.train, split.valid, split.test, mode=args.weight_mode
    )

    feature_cols = build_feature_columns(
        model_df,
        label_cols=["event", "delta"],
        aux_cols=["icu_mortality"],
        extra_drop=DEFAULT_EXTRA_DROP,
    )

    train_path = out_dir / "train_df.csv"
    valid_path = out_dir / "valid_df.csv"
    test_path = out_dir / "test_df.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    valid_df.to_csv(valid_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    metadata = {
        "config": vars(cfg),
        "weight_mode": args.weight_mode,
        "pos_weight": pos_weight,
        "feature_cols": feature_cols,
        "selected_files": {k: str(v) for k, v in selected.items()},
        "train_summary": summarize_label_balance(train_df),
        "valid_summary": summarize_label_balance(valid_df),
        "test_summary": summarize_label_balance(test_df),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[DONE] preprocessing complete")
    print(f" - {train_path}")
    print(f" - {valid_path}")
    print(f" - {test_path}")
    print(f" - {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
