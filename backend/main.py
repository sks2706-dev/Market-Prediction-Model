import math
import os
import traceback
from datetime import date, datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from features import derived_features, original_features_from_yfinance

app = FastAPI(title="Stock Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "models", "clf_knn.joblib")
SCALER_PATH = os.path.join(ROOT_DIR, "models", "feature_scaler.joblib")

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

EARLIEST_DATE = date(2010, 7, 21)

FEATURE_COLUMNS = [
    "Daily_Return", "Intraday_Return", "Daily_Range",
    "Return_Lag1", "Return_Lag2", "Return_Lag3", "Return_Lag4", "Return_Lag5",
    "Return_5D", "Return_10D", "RSI_14",
    "Close_to_MA10", "Close_to_MA50", "MA10_to_MA50",
    "Volatility10", "Volatility30", "Vol_Ratio_10_30",
    "Turnover_to_MA10", "Volume_to_MA10", "Volume_Lag1",
    "Close_to_VWAP", "Close_Location", "Upper_Shadow_Ratio", "Lower_Shadow_Ratio"
]


def safe_float(val, default=0.0, decimals=2):
    """Prevents JSON serialization errors by converting NaN / Inf to standard floats."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, decimals)
    except (ValueError, TypeError):
        return default


@app.get("/api/predict")
def predict_stock(target_date: str = Query(..., description="Date in YYYY-MM-DD format (min: 2010-07-21)")):
    try:
        try:
            parsed_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

        if parsed_date < EARLIEST_DATE or parsed_date > date.today():
            raise HTTPException(status_code=400, detail=f"Date must be between {EARLIEST_DATE} and today.")

        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model or Scaler artifact missing.")

        # 240 back and 130 front, capped at present date
        start_fetch = (parsed_date - pd.Timedelta(days=240)).strftime("%Y-%m-%d")
        capped_end = min(parsed_date + pd.Timedelta(days=130), date.today())
        end_fetch = (capped_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        df_raw = yf.download("TATACONSUM.NS", start=start_fetch, end=end_fetch, interval="1d", auto_adjust=False, actions=False)

        if df_raw.empty:
            raise HTTPException(status_code=404, detail=f"No market data found for {target_date}.")

        if isinstance(df_raw.columns, pd.MultiIndex):
            if 'TATACONSUM.NS' in df_raw.columns.levels[0]:
                df_raw.columns = df_raw.columns.droplevel(0)
            else:
                df_raw.columns = df_raw.columns.droplevel(1)

        df_raw = df_raw[~df_raw.index.duplicated(keep='first')]

        df_base = original_features_from_yfinance(df_raw)
        df_derived = derived_features(df_base)

        if 'Date' not in df_base.columns and isinstance(df_base.index, pd.DatetimeIndex):
            df_base = df_base.reset_index()

        df_base['Date_Str'] = pd.to_datetime(df_base['Date']).dt.strftime('%Y-%m-%d')
        target_date_str = parsed_date.strftime('%Y-%m-%d')
        
        valid_indices = df_base[df_base['Date_Str'] <= target_date_str].index

        if len(valid_indices) == 0:
            target_pos = 0
        else:
            target_pos = int(df_base.index.get_loc(valid_indices[-1]))

        latest_base_row = df_base.iloc[target_pos].to_dict()
        matched_target_date = latest_base_row['Date']

        if 'Date' in df_derived.columns:
            target_derived_row = df_derived[df_derived['Date'] == matched_target_date]
        else:
            target_derived_row = df_derived.loc[[df_base.index[target_pos]]]

        if target_derived_row.empty:
            raise HTTPException(status_code=400, detail="Insufficient historical context to compute technical features for this date.")

        raw_feature_df = target_derived_row[FEATURE_COLUMNS].copy()
        raw_feature_df = raw_feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        current_close = safe_float(latest_base_row.get('Close', 0.0))

        scaled_feature_vector = scaler.transform(raw_feature_df)
        predicted_class = int(model.predict(scaled_feature_vector)[0])

        confidence = 75.0
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_feature_vector)[0]
            confidence = safe_float(probabilities[predicted_class] * 100, default=75.0, decimals=1)

        # Chart view framing (90 days back and 90 ahead)
        start_chart_pos = max(0, target_pos - 90)
        end_chart_pos = min(len(df_base), target_pos + 91)
        chart_df = df_base.iloc[start_chart_pos:end_chart_pos]

        chart_data = [
            {
                "time": pd.to_datetime(row['Date']).strftime('%Y-%m-%d'),
                "value": safe_float(row.get('Close', 0.0))
            }
            for _, row in chart_df.iterrows()
            if not pd.isna(row.get('Close'))
        ]

        formatted_original_features = {
            "Open": safe_float(latest_base_row.get("Open", 0.0)),
            "High": safe_float(latest_base_row.get("High", 0.0)),
            "Low": safe_float(latest_base_row.get("Low", 0.0)),
            "Close": safe_float(latest_base_row.get("Close", 0.0)),
            "Last": safe_float(latest_base_row.get("Last", latest_base_row.get("Close", 0.0))),
        }

        data_date_used = pd.to_datetime(latest_base_row['Date']).strftime('%Y-%m-%d')

        return {
            "symbol": "TATACONSUM.NS",
            "requested_date": target_date,
            "data_date_used": data_date_used,
            "current_price": current_close,
            "predicted_class": predicted_class,
            "signal": "UP" if predicted_class == 1 else "DOWN",
            "confidence": confidence,
            "chart_data": chart_data,
            "original_features": formatted_original_features
        }

    except Exception as e:
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))