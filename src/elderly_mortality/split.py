from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def split_by_subject_stratified(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    stay_col: str = "stay_id",
    stratify_label_col: str = "icu_mortality",
    random_state: int = 42,
    test_size: float = 0.20,
    valid_size_within_train: float = 0.20,
) -> SplitData:
    required = {subject_col, stay_col, stratify_label_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    subj_label = (
        df[[subject_col, stratify_label_col]]
        .drop_duplicates()
        .groupby(subject_col)[stratify_label_col]
        .max()
        .reset_index()
        .rename(columns={stratify_label_col: "_subj_label"})
    )

    train_subj, test_subj = train_test_split(
        subj_label[subject_col].values,
        test_size=test_size,
        random_state=random_state,
        stratify=subj_label["_subj_label"].values,
    )

    train_subj, valid_subj = train_test_split(
        train_subj,
        test_size=valid_size_within_train,
        random_state=random_state,
        stratify=subj_label.set_index(subject_col).loc[train_subj, "_subj_label"].values,
    )

    train_df = df[df[subject_col].isin(train_subj)].copy()
    valid_df = df[df[subject_col].isin(valid_subj)].copy()
    test_df = df[df[subject_col].isin(test_subj)].copy()

    _assert_disjoint(train_df, valid_df, subject_col, "train", "valid")
    _assert_disjoint(train_df, test_df, subject_col, "train", "test")
    _assert_disjoint(valid_df, test_df, subject_col, "valid", "test")
    _assert_disjoint(train_df, valid_df, stay_col, "train", "valid")
    _assert_disjoint(train_df, test_df, stay_col, "train", "test")
    _assert_disjoint(valid_df, test_df, stay_col, "valid", "test")

    return SplitData(train=train_df, valid=valid_df, test=test_df)


def _assert_disjoint(a: pd.DataFrame, b: pd.DataFrame, col: str, name_a: str, name_b: str) -> None:
    overlap = set(a[col].unique()) & set(b[col].unique())
    if overlap:
        sample = list(overlap)[:10]
        raise ValueError(f"Leakage detected: {name_a}/{name_b} overlap in '{col}'. sample={sample}")


def add_sample_weight_by_event(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    event_col: str = "event",
    mode: str = "ratio",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    pos = float((train_df[event_col] == 1).sum())
    neg = float((train_df[event_col] == 0).sum())
    if pos == 0:
        raise ValueError("No positive event rows in train split.")

    if mode == "ratio":
        pos_weight = neg / pos
    elif mode == "sqrt_ratio":
        pos_weight = float(np.sqrt(neg / pos))
    else:
        raise ValueError("mode must be one of {'ratio', 'sqrt_ratio'}")

    def _attach(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["sample_weight"] = np.where(out[event_col] == 1, pos_weight, 1.0).astype("float32")
        return out

    return _attach(train_df), _attach(valid_df), _attach(test_df), float(pos_weight)

