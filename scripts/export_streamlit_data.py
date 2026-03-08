from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elderly_mortality.streamlit_export import export_streamlit_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export streamlit-ready sample_patient_header.csv and sample_timeseries.csv")
    p.add_argument("--source-csv", required=True, help="Source table CSV (e.g., test_df.csv)")
    p.add_argument("--out-dir", default=str(ROOT / "streamlit" / "data"), help="Output streamlit data directory")
    p.add_argument("--n-patients", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--risk-scores-csv",
        default=None,
        help="Optional row-level risk score CSV. Must contain either 'score' or 'risk_score'.",
    )
    return p.parse_args()


def attach_risk_scores(df: pd.DataFrame, risk_csv: str) -> pd.DataFrame:
    risk_df = pd.read_csv(risk_csv)
    if "risk_score" in risk_df.columns:
        score_col = "risk_score"
    elif "score" in risk_df.columns:
        score_col = "score"
    else:
        raise KeyError("risk-scores-csv must contain 'score' or 'risk_score'.")

    out = df.copy()
    if {"stay_id", "t"}.issubset(risk_df.columns) and {"stay_id", "t"}.issubset(out.columns):
        merged = out.merge(
            risk_df[["stay_id", "t", score_col]].rename(columns={score_col: "risk_score"}),
            on=["stay_id", "t"],
            how="left",
        )
        merged["risk_score"] = pd.to_numeric(merged["risk_score"], errors="coerce").fillna(0.0)
        return merged

    if len(risk_df) != len(out):
        raise ValueError(
            "Length mismatch between source-csv and risk-scores-csv, and no (stay_id,t) keys found for merge."
        )
    out["risk_score"] = pd.to_numeric(risk_df[score_col], errors="coerce").fillna(0.0).to_numpy()
    return out


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.source_csv)
    if args.risk_scores_csv:
        source = attach_risk_scores(source, args.risk_scores_csv)

    hd_file, ts_file = export_streamlit_bundle(
        source,
        out_dir=args.out_dir,
        n_patients=args.n_patients,
        seed=args.seed,
    )
    print("[DONE] streamlit data exported")
    print(f" - {hd_file}")
    print(f" - {ts_file}")


if __name__ == "__main__":
    main()

