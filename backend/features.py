import pandas as pd
import numpy as np

def original_features_from_yfinance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'Date' not in df.columns:
        df = df.reset_index()

    # Flatten multi-index headers if returned
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    df['Last'] = df['Close']
    df['Total Trade Quantity'] = df['Volume']

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['Turnover (Lacs)'] = (df['Total Trade Quantity'] * typical_price) / 100000

    # Format numeric precision
    df['Open'] = df['Open'].round(2)
    df['High'] = df['High'].round(2)
    df['Low'] = df['Low'].round(2)
    df['Last'] = df['Last'].round(2)
    df['Close'] = df['Close'].round(2)
    df['Turnover (Lacs)'] = df['Turnover (Lacs)'].round(2)
    df['Total Trade Quantity'] = df['Total Trade Quantity'].astype(int)

    target_columns = [
        'Date', 
        'Open', 
        'High', 
        'Low', 
        'Last', 
        'Close', 
        'Total Trade Quantity', 
        'Turnover (Lacs)'
    ]

    return df[target_columns]


def derived_features(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Takes the base CSV DataFrame and generates all relative indicators & signals.
    """
    df = df_base.copy()

    # Base Returns & Lagged Signals
    df["Daily_Return"] = 100 * (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["Intraday_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Daily_Range"] = (df["High"] - df["Low"]) / df["Low"]

    df["Return_Lag1"] = df["Daily_Return"].shift(1)
    df["Return_Lag2"] = df["Daily_Return"].shift(2)
    df["Return_Lag3"] = df["Daily_Return"].shift(3)
    df["Return_Lag4"] = df["Daily_Return"].shift(4)
    df["Return_Lag5"] = df["Daily_Return"].shift(5)

    # Multi-Horizon Momentum & RSI
    df["Return_5D"] = 100 * (df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5)
    df["Return_10D"] = 100 * (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    # Moving Average Ratios (Trend Alignment)
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

    return df