from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception as exc:  # pragma: no cover
    raise ImportError("TensorFlow is required for LSTM utilities.") from exc


PAD_VALUE = -999.0


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_sequences(
    df_long: pd.DataFrame,
    feature_cols: list[str],
    *,
    id_col: str = "stay_id",
    time_col: str = "t",
    label_col: str = "_future_label",
) -> tuple[pd.DataFrame, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    df_sorted = df_long.sort_values([id_col, time_col]).copy()
    x_list, y_list, idx_list = [], [], []
    for _, g in df_sorted.groupby(id_col, sort=False):
        idx = g.index.to_numpy()
        x = g[feature_cols].to_numpy(dtype=np.float32)
        y = g[label_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        x_list.append(x)
        y_list.append(y)
        idx_list.append(idx)
    return df_sorted, x_list, y_list, idx_list


def pad_3d(x_list: list[np.ndarray], max_len: int, pad_value: float = PAD_VALUE) -> tuple[np.ndarray, np.ndarray]:
    n_features = x_list[0].shape[1]
    x_pad = np.full((len(x_list), max_len, n_features), pad_value, dtype=np.float32)
    mask = np.zeros((len(x_list), max_len), dtype=bool)
    for i, x in enumerate(x_list):
        length = min(x.shape[0], max_len)
        x_pad[i, :length, :] = x[:length]
        mask[i, :length] = True
    return x_pad, mask


def pad_y(y_list: list[np.ndarray], max_len: int) -> np.ndarray:
    y_pad = np.zeros((len(y_list), max_len, 1), dtype=np.float32)
    for i, y in enumerate(y_list):
        length = min(y.shape[0], max_len)
        y_pad[i, :length, :] = y[:length]
    return y_pad


def build_lstm_timewise(input_shape: tuple[int, int], *, pad_value: float = PAD_VALUE) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=pad_value)(inp)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(1, activation="sigmoid", dtype="float32"))(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_lstm_row_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    id_col: str = "stay_id",
    time_col: str = "t",
    label_col: str = "_future_label",
    max_len: int = 120,
    epochs: int = 50,
    batch_size: int = 256,
    patience: int = 8,
    seed: int = 42,
) -> tuple[tf.keras.Model, StandardScaler]:
    set_seed(seed)

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float32))

    tr = train_df.copy()
    va = valid_df.copy()
    tr[feature_cols] = scaler.transform(tr[feature_cols].to_numpy(dtype=np.float32))
    va[feature_cols] = scaler.transform(va[feature_cols].to_numpy(dtype=np.float32))

    _, x_tr_list, y_tr_list, _ = make_sequences(
        tr, feature_cols, id_col=id_col, time_col=time_col, label_col=label_col
    )
    _, x_va_list, y_va_list, _ = make_sequences(
        va, feature_cols, id_col=id_col, time_col=time_col, label_col=label_col
    )

    x_tr, m_tr = pad_3d(x_tr_list, max_len)
    x_va, m_va = pad_3d(x_va_list, max_len)
    y_tr = pad_y(y_tr_list, max_len)
    y_va = pad_y(y_va_list, max_len)
    sw_tr = m_tr.astype(np.float32)
    sw_va = m_va.astype(np.float32)

    model = build_lstm_timewise((max_len, len(feature_cols)))
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=patience,
        restore_best_weights=True,
        verbose=0,
    )
    model.fit(
        x_tr,
        y_tr,
        validation_data=(x_va, y_va, sw_va),
        sample_weight=sw_tr,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1,
    )
    return model, scaler


def predict_row_score_lstm(
    model: tf.keras.Model,
    scaler: StandardScaler,
    df_eval: pd.DataFrame,
    feature_cols: list[str],
    *,
    id_col: str = "stay_id",
    time_col: str = "t",
    label_col: str = "_future_label",
    max_len: int = 120,
) -> np.ndarray:
    if not df_eval.index.is_unique:
        raise ValueError("df_eval index must be unique for safe remapping.")

    df_scaled = df_eval.copy()
    df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols].to_numpy(dtype=np.float32))
    df_sorted, x_list, _, idx_list = make_sequences(
        df_scaled, feature_cols, id_col=id_col, time_col=time_col, label_col=label_col
    )

    x_pad, _ = pad_3d(x_list, max_len)
    yhat = model.predict(x_pad, verbose=0).reshape(len(x_list), max_len)

    pos = {idx: i for i, idx in enumerate(df_eval.index.to_numpy())}
    row_score = np.full((len(df_eval),), np.nan, dtype=np.float32)
    for i, idx in enumerate(idx_list):
        length = min(len(idx), max_len)
        for j in range(length):
            row_score[pos[idx[j]]] = yhat[i, j]

    if np.isnan(row_score).any():
        raise RuntimeError("NaN detected while remapping LSTM row scores.")
    _ = df_sorted  # keep explicit for readability; no side effect.
    return row_score

