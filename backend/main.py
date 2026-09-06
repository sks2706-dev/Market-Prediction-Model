from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, date
import yfinance as yf
import pandas as pd
import joblib
import os

from features import original_features_from_yfinance, derived_features

app = FastAPI(title="TATACONSUM Stock Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "clf_knn.joblib")
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

EARLIEST_DATE = date(2010, 7, 21)

FEATURE_COLUMNS = [
    "Daily_Return", "Intraday_Return", "Daily_Range", 
    "Return_Lag1", "Return_Lag2", "Return_Lag3", "Return_Lag4", "Return_Lag5", 
    "Return_5D", "Return_10D", "RSI_14", 
    "Close_to_MA10", "Close_to_MA50", "MA10_to_MA50", 
    "Volatility10", "Volatility30", "Vol_Ratio_10_30", 
    "Turnover_to_MA10", "Volume_to_MA10", "Volume_Lag1", "Close_to_VWAP", 
    "Close_Location", "Upper_Shadow_Ratio", "Lower_Shadow_Ratio"
]

@app.get("/api/predict")
def predict_stock(target_date: str = Query(..., description="Date in YYYY-MM-DD format (min: 2010-07-21)")):
    try:
        try:
            parsed_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

        if parsed_date < EARLIEST_DATE or parsed_date > date.today():
            raise HTTPException(status_code=400, detail=f"Date must be between {EARLIEST_DATE} and today.")

        # 120-day buffer
        start_fetch = (parsed_date - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
        end_fetch = (parsed_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        df_raw = yf.download("TATACONSUM.NS", start=start_fetch, end=end_fetch, interval="1d", auto_adjust=False, actions=False)

        if df_raw.empty:
            raise HTTPException(status_code=404, detail=f"No market data found for {target_date}.")

        # Step 1: Base features matching original CSV
        df_base = original_features_from_yfinance(df_raw)

        # Step 2: Derived technical features from base
        df_derived = derived_features(df_base)

        # Grab latest row
        latest_base_row = df_base.iloc[-1].to_dict()
        latest_derived_row = df_derived.iloc[-1]
        
        current_close = float(latest_base_row['Close'])

        # Prepare feature vector for KNN prediction
        feature_vector = [float(latest_derived_row[col]) for col in FEATURE_COLUMNS]

        predicted_class = int(model.predict([feature_vector])[0])
        
        return {
            "symbol": "TATACONSUM.NS",
            "requested_date": target_date,
            "data_date_used": latest_base_row['Date'],
            "current_price": current_close,
            "predicted_class": predicted_class,
            "signal": "UP" if predicted_class == 1 else "DOWN",
            "original_features": latest_base_row
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))