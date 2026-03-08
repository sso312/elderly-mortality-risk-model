from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .config import AGG_RULE_ROWS, BOUNDS_ROWS, MAPPING_ROWS


def _pick_csv(data_root: Path, pattern: str) -> Path:
    hits = sorted(data_root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"Missing CSV for pattern '{pattern}' under '{data_root}'.")
    return hits[0]


def register_raw_views(con: duckdb.DuckDBPyConnection, data_root: str | Path) -> dict[str, Path]:
    root = Path(data_root)
    mapping = {
        "icustays": "*icustays*.csv",
        "chartevents": "*chartevents*.csv",
        "labevents": "*labevents*.csv",
        "admissions": "*admissions*.csv",
        "patients": "*patients*.csv",
    }
    selected = {name: _pick_csv(root, pat) for name, pat in mapping.items()}

    for name, path in selected.items():
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT * FROM read_csv_auto('{path.as_posix()}', header=true);
            """
        )
    return selected


def build_elderly_cohort(
    con: duckdb.DuckDBPyConnection,
    *,
    max_hours: int = 120,
    min_age: int = 65,
) -> pd.DataFrame:
    sql = f"""
    WITH
    adm AS (
      SELECT
        subject_id,
        hadm_id,
        TRY_CAST(admittime AS TIMESTAMP) AS admittime,
        TRY_CAST(dischtime AS TIMESTAMP) AS dischtime,
        TRY_CAST(deathtime AS TIMESTAMP) AS deathtime
      FROM admissions
    ),
    pat AS (
      SELECT
        subject_id,
        anchor_age,
        anchor_year,
        TRY_CAST(dod AS TIMESTAMP) AS dod
      FROM patients
    ),
    icu0 AS (
      SELECT
        subject_id,
        hadm_id,
        stay_id,
        TRY_CAST(intime AS TIMESTAMP) AS intime,
        TRY_CAST(outtime AS TIMESTAMP) AS outtime
      FROM icustays
    ),
    adm_age AS (
      SELECT
        a.subject_id,
        a.hadm_id,
        a.dischtime,
        a.deathtime,
        p.anchor_age,
        p.anchor_year,
        (p.anchor_age + (EXTRACT(YEAR FROM a.admittime) - p.anchor_year))::INTEGER AS age
      FROM adm a
      LEFT JOIN pat p ON a.subject_id = p.subject_id
    ),
    adm65 AS (
      SELECT *
      FROM adm_age
      WHERE age >= {int(min_age)}
    ),
    icu_join AS (
      SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime,
        i.outtime,
        a.dischtime,
        a.age,
        p.dod,
        a.deathtime,
        COALESCE(a.deathtime, p.dod) AS death_ts
      FROM icu0 i
      INNER JOIN adm65 a
        ON i.subject_id = a.subject_id
       AND i.hadm_id = a.hadm_id
      LEFT JOIN pat p
        ON i.subject_id = p.subject_id
    ),
    icu_labeled AS (
      SELECT
        *,
        CASE
          WHEN death_ts IS NOT NULL
           AND intime IS NOT NULL
           AND outtime IS NOT NULL
           AND death_ts >= intime
           AND death_ts <= outtime
          THEN 1 ELSE 0
        END AS icu_mortality
      FROM icu_join
    ),
    icu_features AS (
      SELECT
        subject_id,
        hadm_id,
        stay_id,
        intime,
        outtime,
        age,
        icu_mortality,
        CASE
          WHEN icu_mortality = 1
          THEN (DATE_DIFF('second', intime, death_ts) / 3600.0)
          ELSE NULL
        END AS death_hour,
        LEAST((DATE_DIFF('second', intime, outtime) / 3600.0), {int(max_hours)}) AS icu_los_hours_clipped
      FROM icu_labeled
    )
    SELECT
      subject_id, hadm_id, stay_id,
      intime, outtime,
      age, icu_los_hours_clipped,
      icu_mortality, death_hour
    FROM icu_features
    """
    cohort = con.execute(sql).df()
    con.register("cohort", cohort)
    return cohort


def register_mapping_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS mapping;")
    con.execute(
        """
        CREATE TABLE mapping(
          variable VARCHAR,
          src_table VARCHAR,
          itemid INTEGER,
          unit VARCHAR
        );
        """
    )
    con.executemany("INSERT INTO mapping VALUES (?, ?, ?, ?)", MAPPING_ROWS)

    con.execute("DROP TABLE IF EXISTS agg_rule;")
    con.execute("CREATE TABLE agg_rule(variable VARCHAR, agg VARCHAR);")
    con.executemany("INSERT INTO agg_rule VALUES (?, ?)", AGG_RULE_ROWS)

    con.execute("DROP TABLE IF EXISTS bounds;")
    con.execute("CREATE TABLE bounds(variable VARCHAR, min_val DOUBLE, max_val DOUBLE);")
    con.executemany("INSERT INTO bounds VALUES (?, ?, ?)", BOUNDS_ROWS)


def build_long_hourly(con: duckdb.DuckDBPyConnection, *, max_hours: int = 120) -> None:
    sql = f"""
    CREATE OR REPLACE TABLE long_hourly AS
    WITH
    co AS (
      SELECT
        stay_id,
        subject_id,
        hadm_id,
        CAST(intime AS TIMESTAMP) AS intime,
        CAST(outtime AS TIMESTAMP) AS outtime
      FROM cohort
    ),
    ce AS (
      SELECT
        c.stay_id,
        TRY_CAST(ce.charttime AS TIMESTAMP) AS time,
        m.variable,
        TRY_CAST(ce.valuenum AS DOUBLE) AS value_raw
      FROM chartevents ce
      JOIN co c USING (stay_id)
      JOIN mapping m
        ON m.src_table = 'chartevents' AND m.itemid = ce.itemid
      WHERE ce.valuenum IS NOT NULL
    ),
    lab AS (
      SELECT
        c.stay_id,
        TRY_CAST(l.charttime AS TIMESTAMP) AS time,
        m.variable,
        TRY_CAST(l.valuenum AS DOUBLE) AS value_raw
      FROM labevents l
      JOIN co c
        ON c.subject_id = l.subject_id
       AND c.hadm_id = l.hadm_id
      JOIN mapping m
        ON m.src_table = 'labevents' AND m.itemid = l.itemid
      WHERE l.valuenum IS NOT NULL
        AND TRY_CAST(l.charttime AS TIMESTAMP) BETWEEN c.intime AND c.outtime
    ),
    events AS (
      SELECT * FROM ce
      UNION ALL
      SELECT * FROM lab
    ),
    bucketed AS (
      SELECT
        e.stay_id,
        CAST(FLOOR(DATE_DIFF('second', c.intime, e.time) / 3600.0) AS INTEGER) AS t,
        e.variable,
        CASE
          WHEN e.variable = 'Temp' AND e.value_raw > 45 THEN (e.value_raw - 32) * 5.0 / 9.0
          ELSE e.value_raw
        END AS value
      FROM events e
      JOIN co c USING (stay_id)
      WHERE e.time IS NOT NULL
    ),
    windowed AS (
      SELECT * FROM bucketed
      WHERE t >= 0 AND t < {int(max_hours)}
    ),
    cleaned AS (
      SELECT
        w.stay_id,
        w.t,
        w.variable,
        CASE
          WHEN b.variable IS NULL THEN w.value
          WHEN w.variable = 'SysBP' AND (w.value <= 0 OR w.value > b.max_val) THEN NULL
          WHEN w.value < b.min_val OR w.value > b.max_val THEN NULL
          ELSE w.value
        END AS value_clean,
        CASE
          WHEN b.variable IS NULL THEN 1
          WHEN w.variable = 'SysBP' AND (w.value <= 0 OR w.value > b.max_val) THEN 0
          WHEN w.value < b.min_val OR w.value > b.max_val THEN 0
          ELSE 1
        END AS measured_clean
      FROM windowed w
      LEFT JOIN bounds b USING (variable)
    ),
    agg_final AS (
      SELECT
        c.stay_id,
        c.t,
        c.variable,
        CASE
          WHEN r.agg = 'min' THEN MIN(c.value_clean)
          WHEN r.agg = 'max' THEN MAX(c.value_clean)
          ELSE AVG(c.value_clean)
        END AS value,
        MAX(c.measured_clean) AS measured
      FROM cleaned c
      LEFT JOIN agg_rule r USING (variable)
      GROUP BY c.stay_id, c.t, c.variable, r.agg
    )
    SELECT * FROM agg_final;
    """
    con.execute(sql)


def _pivot_to_wide(con: duckdb.DuckDBPyConnection, *, max_hours: int = 120) -> None:
    vars_in_long = [r[0] for r in con.execute("SELECT DISTINCT variable FROM long_hourly ORDER BY 1").fetchall()]
    if not vars_in_long:
        raise RuntimeError("No variables found in long_hourly.")

    value_cols_sql = ",\n    ".join(
        [f"MAX(CASE WHEN variable='{v}' THEN value END) AS {v}" for v in vars_in_long]
    )
    meas_cols_sql = ",\n    ".join(
        [f"MAX(CASE WHEN variable='{v}' THEN measured END) AS {v}_measured" for v in vars_in_long]
    )

    sql = f"""
    CREATE OR REPLACE TABLE wide_hourly_raw AS
    WITH pivoted AS (
      SELECT
        stay_id,
        t,
        {value_cols_sql},
        {meas_cols_sql}
      FROM long_hourly
      GROUP BY stay_id, t
    ),
    grid AS (
      SELECT s.stay_id, g.t
      FROM (SELECT DISTINCT stay_id FROM long_hourly) s
      CROSS JOIN generate_series(0, {int(max_hours) - 1}) g(t)
    )
    SELECT
      g.stay_id,
      g.t,
      p.*
    EXCLUDE (stay_id, t)
    FROM grid g
    LEFT JOIN pivoted p
      ON g.stay_id = p.stay_id
     AND g.t = p.t
    ORDER BY g.stay_id, g.t;
    """
    con.execute(sql)


def build_wide_hourly_ffill(con: duckdb.DuckDBPyConnection, *, max_hours: int = 120) -> pd.DataFrame:
    _pivot_to_wide(con, max_hours=max_hours)
    wide_df = con.execute("SELECT * FROM wide_hourly_raw ORDER BY stay_id, t").df()

    value_cols = [c for c in wide_df.columns if c not in {"stay_id", "t"} and not c.endswith("_measured")]
    for c in value_cols:
        meas_col = f"{c}_measured"
        if meas_col not in wide_df.columns:
            wide_df[meas_col] = wide_df[c].notna().astype(int)

    wide_df = wide_df.sort_values(["stay_id", "t"])
    wide_df[value_cols] = wide_df.groupby("stay_id", sort=False)[value_cols].ffill()
    return wide_df


def attach_event_labels(
    wide_hourly_ffill: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    max_hours: int = 120,
) -> pd.DataFrame:
    out = wide_hourly_ffill.copy()
    event_time = cohort[["stay_id", "icu_mortality", "death_hour"]].copy()
    event_time["t_event"] = event_time["death_hour"].apply(
        lambda x: int(x // 1) if pd.notna(x) else None
    )
    out = out.merge(event_time[["stay_id", "icu_mortality", "t_event"]], on="stay_id", how="left")

    valid_event = (
        (out["icu_mortality"] == 1)
        & out["t_event"].notna()
        & (out["t_event"] >= 0)
        & (out["t_event"] < int(max_hours))
    )
    out["event"] = ((out["t"] == out["t_event"]) & valid_event).astype(int)
    out["delta"] = out["event"].astype(int)
    return out


def build_full_model_table(
    con: duckdb.DuckDBPyConnection,
    *,
    max_hours: int = 120,
    min_age: int = 65,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = build_elderly_cohort(con, max_hours=max_hours, min_age=min_age)
    register_mapping_tables(con)
    build_long_hourly(con, max_hours=max_hours)
    wide = build_wide_hourly_ffill(con, max_hours=max_hours)
    model_df = attach_event_labels(wide, cohort, max_hours=max_hours)
    model_df = model_df.merge(
        cohort[["stay_id", "subject_id", "hadm_id", "icu_mortality", "death_hour", "icu_los_hours_clipped"]],
        on="stay_id",
        how="left",
        suffixes=("", "_cohort"),
    )
    if "icu_mortality_cohort" in model_df.columns:
        model_df["icu_mortality"] = model_df["icu_mortality_cohort"].fillna(model_df["icu_mortality"])
        model_df = model_df.drop(columns=["icu_mortality_cohort"])
    return model_df, cohort


def summarize_label_balance(df: pd.DataFrame, event_col: str = "event") -> dict[str, Any]:
    n_total = int(len(df))
    n_event = int(df[event_col].sum())
    return {
        "n_rows": n_total,
        "n_event_rows": n_event,
        "n_nonevent_rows": n_total - n_event,
        "event_rate": float(n_event / n_total) if n_total else 0.0,
    }
