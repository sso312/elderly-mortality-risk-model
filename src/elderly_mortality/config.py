from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


MAPPING_ROWS: list[tuple[str, str, int, str]] = [
    ("HeartRate", "chartevents", 220045, "bpm"),
    ("HeartRate", "chartevents", 211, "bpm"),
    ("RespRate", "chartevents", 220210, "breaths/min"),
    ("RespRate", "chartevents", 224690, "breaths/min"),
    ("SpO2", "chartevents", 220277, "percent"),
    ("SysBP", "chartevents", 220179, "mmHg"),
    ("SysBP", "chartevents", 51, "mmHg"),
    ("SysBP", "chartevents", 455, "mmHg"),
    ("DiasBP", "chartevents", 220180, "mmHg"),
    ("DiasBP", "chartevents", 8368, "mmHg"),
    ("MeanBP", "chartevents", 220181, "mmHg"),
    ("MeanBP", "chartevents", 52, "mmHg"),
    ("Temp", "chartevents", 223761, "C"),
    ("Temp", "chartevents", 678, "C"),
    ("Weight", "chartevents", 226512, "kg"),
    ("Weight", "chartevents", 763, "kg"),
    ("Height", "chartevents", 226707, "cm"),
    ("Height", "chartevents", 226730, "cm"),
    ("FiO2", "chartevents", 223835, "ratio"),
    ("FiO2", "chartevents", 3420, "ratio"),
    ("GCS_Eye", "chartevents", 220739, "score"),
    ("GCS_Verbal", "chartevents", 223900, "score"),
    ("GCS_Motor", "chartevents", 223901, "score"),
    ("CapillaryRefill", "chartevents", 223951, "sec"),
    ("Glucose", "chartevents", 220621, "mg/dL"),
    ("Glucose", "labevents", 50931, "mg/dL"),
    ("Glucose", "labevents", 50809, "mg/dL"),
    ("pH", "labevents", 50820, "pH"),
    ("pH", "labevents", 50818, "pH"),
]

AGG_RULE_ROWS: list[tuple[str, str]] = [
    ("SpO2", "min"),
    ("pH", "min"),
    ("GCS_Eye", "min"),
    ("GCS_Verbal", "min"),
    ("GCS_Motor", "min"),
    ("HeartRate", "max"),
    ("RespRate", "max"),
    ("Temp", "max"),
    ("Glucose", "max"),
    ("SysBP", "max"),
    ("DiasBP", "max"),
    ("MeanBP", "max"),
    ("Weight", "max"),
    ("Height", "max"),
]

BOUNDS_ROWS: list[tuple[str, float, float]] = [
    ("SpO2", 0, 100),
    ("pH", 6.8, 8.0),
    ("Temp", 30, 45),
    ("HeartRate", 0, 300),
    ("SysBP", 0, 300),
    ("DiasBP", 0, 200),
    ("MeanBP", 0, 200),
    ("RespRate", 0, 60),
    ("Glucose", 0, 2000),
    ("GCS_Eye", 1, 4),
    ("GCS_Verbal", 1, 5),
    ("GCS_Motor", 1, 6),
]

TREND_VARIABLES: list[str] = [
    "HeartRate",
    "MeanBP",
    "SpO2",
    "RespRate",
    "GCS_Total",
    "Temp",
    "Glucose",
    "SysBP",
    "DiasBP",
    "pH",
]

DEFAULT_EXTRA_DROP: set[str] = {
    "calc_DiasBP",
    "ShockIndex",
    "PulsePressure",
    "ModShockIndex",
    "ROX_Index",
    "_label_observable",
}


@dataclass
class PreprocessConfig:
    max_hours: int = 120
    min_age: int = 65
    random_state: int = 42
    train_size: float = 0.64
    valid_size: float = 0.16
    test_size: float = 0.20
    trend_variables: Sequence[str] = field(default_factory=lambda: TREND_VARIABLES)


@dataclass
class HorizonConfig:
    horizon_hours: int = 6
    cutoff_hours: int = 24
    target_recall: float = 0.80
    drop_after_event: bool = True
    apply_cutoff_to_train: bool = False

