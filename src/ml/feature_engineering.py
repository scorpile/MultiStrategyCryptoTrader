"""Feature engineering helpers for ML models."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import pandas as pd


def ohlcv_to_dataframe(ohlcv_rows: Iterable[Dict[str, float]]) -> pd.DataFrame:
    """Convert a list of OHLCV dicts into a pandas DataFrame with datetime index."""
    df = pd.DataFrame(list(ohlcv_rows))
    if df.empty:
        return df
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.set_index("open_time")
    return df.astype(float, errors="ignore")


def generate_features(
    ohlcv_rows: Iterable[Dict[str, float]],
    *,
    include_returns: bool = True,
) -> pd.DataFrame:
    """Return a feature matrix derived from OHLCV rows."""
    df = ohlcv_to_dataframe(ohlcv_rows).copy()
    if df.empty:
        return df

    df["close_return_1"] = df["close"].pct_change()
    df["close_return_5"] = df["close"].pct_change(5)
    df["rolling_mean_5"] = df["close"].rolling(window=5).mean()
    df["rolling_mean_20"] = df["close"].rolling(window=20).mean()
    df["rolling_std_20"] = df["close"].rolling(window=20).std()
    df["volume_zscore_20"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_position_20"] = (df["close"] - df["rolling_mean_20"]) / (df["rolling_std_20"] + 1e-9)
    
    # Additional scalping features
    df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
    df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"]
    df["momentum_3"] = df["close"] - df["close"].shift(3)
    df["momentum_5"] = df["close"] - df["close"].shift(5)
    df["volume_ratio_5"] = df["volume"] / df["volume"].rolling(5).mean()

    feature_cols = [
        "close_return_1",
        "close_return_5",
        "rolling_mean_5",
        "rolling_mean_20",
        "rolling_std_20",
        "volume_zscore_20",
        "high_low_range",
        "close_position_20",
        "vwap_deviation",
        "momentum_3",
        "momentum_5",
        "volume_ratio_5",
    ]
    features = df[feature_cols].dropna()
    return features


def generate_target_labels(
    ohlcv_rows: Iterable[Dict[str, float]],
    horizon: int = 1,
) -> pd.Series:
    """Produce binary labels (1 if future close is higher, else 0)."""
    df = ohlcv_to_dataframe(ohlcv_rows).copy()
    if df.empty:
        return pd.Series(dtype=int)
    future_close = df["close"].shift(-horizon)
    labels = (future_close > df["close"]).astype(int)
    return labels.loc[df.index]


def build_dataset(
    ohlcv_rows: Iterable[Dict[str, float]],
    *,
    horizon: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return aligned features and labels ready for model training."""
    features = generate_features(ohlcv_rows)
    labels = generate_target_labels(ohlcv_rows, horizon=horizon)
    merged = features.join(labels.rename("label")).dropna()
    return merged.drop(columns=["label"]), merged["label"]
