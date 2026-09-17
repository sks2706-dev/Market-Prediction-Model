"""
Production Training Pipeline for final KNN Model
Input: data/NSE-TATAGLOBAL.csv
Output: models/clf_knn.joblib, models/feature_scaler.joblib
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses raw market data to derived data such that no absolute value is passed"""
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Base Returns & Lagged Signals
    df["Daily_Return"] = 100 * (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["Intraday_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Daily_Range"] = (df["High"] - df["Low"]) / df["Low"]

    for lag in range(1, 6):
        df[f"Return_Lag{lag}"] = df["Daily_Return"].shift(lag)

    # Multi-Horizon Momentum & RSI
    df["Return_5D"] = 100 * (df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5)
    df["Return_10D"] = 100 * (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    # Moving Average Ratios
    df["Close_to_MA10"] = df["Close"] / df["Close"].rolling(10).mean()
    df["Close_to_MA50"] = df["Close"] / df["Close"].rolling(50).mean()
    df["MA10_to_MA50"] = df["Close"].rolling(10).mean() / df["Close"].rolling(50).mean()

    # Volatility & Regime Shifts
    df["Volatility10"] = df["Daily_Return"].rolling(10).std()
    df["Volatility30"] = df["Daily_Return"].rolling(30).std()
    df["Vol_Ratio_10_30"] = df["Volatility10"] / (df["Volatility30"] + 1e-9)

    # Volume & VWAP Ratios
    df["Turnover_to_MA10"] = df["Turnover (Lacs)"] / df["Turnover (Lacs)"].rolling(10).mean()
    df["Volume_to_MA10"] = df["Total Trade Quantity"] / df["Total Trade Quantity"].rolling(10).mean()
    df["Volume_Lag1"] = df["Total Trade Quantity"] / df["Total Trade Quantity"].shift(1)

    df["VWAP"] = (df["Turnover (Lacs)"] * 100000) / df["Total Trade Quantity"]
    df["Close_to_VWAP"] = df["Close"] / df["VWAP"]

    # Candlestick Anatomy
    hl_span = (df["High"] - df["Low"]) + 1e-9
    df["Close_Location"] = (df["Close"] - df["Low"]) / hl_span
    df["Upper_Shadow_Ratio"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / hl_span
    df["Lower_Shadow_Ratio"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / hl_span

    # Target Setup
    df["Target_Return"] = df["Daily_Return"].shift(-1)
    df["Target_Class"] = (df["Target_Return"] > 0).astype(int)

    # Clean missing values resulting from rolling window/lags
    df = df.dropna().reset_index(drop=True)
    return df


def train():

    data_path = "data/NSE-TATAGLOBAL.csv"

    df_raw = pd.read_csv(data_path) #importing data

    df = build_features(df_raw) #adding features

    X = df.drop(
        columns=[
            "Target_Class",
            "Target_Return",
            "Date",
            "Open",
            "High",
            "Low",
            "Last",
            "Close",
            "Total Trade Quantity",
            "Turnover (Lacs)",
            "VWAP",
        ]
    )
    y = df["Target_Class"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.12, shuffle=False) #splitting into x and y

    scaler = StandardScaler().set_output(transform="pandas") # feature scaling
    scaled_X_train = scaler.fit_transform(X_train)

    tscv = TimeSeriesSplit(n_splits=5) # gridsearchcv
    knn_param_grid = {
        "n_neighbors": list(range(1, 61)),
        "p": [1, 1.25, 1.5, 1.75, 2, 3],
        "weights": ["uniform"],
        "metric": ["minkowski", "chebyshev"],
    }
    grid_knn = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=knn_param_grid,
        cv=tscv,
        scoring="f1",
        n_jobs=-1,
    )
    grid_knn.fit(scaled_X_train, y_train)

    best_knn = grid_knn.best_estimator_
    print(f"Best Params: {grid_knn.best_params_}")
    print(f"Best F1 Score: {grid_knn.best_score_:.4f}")

    model_path = "models/clf_knn.joblib"
    scaler_path = "models/feature_scaler.joblib"

    dump(best_knn, model_path) #model export
    dump(scaler, scaler_path)


if __name__ == "__main__":
    train()