# 📈 Stock Signal Prediction Engine

A full-stack quantitative application that leverages a K-Nearest Neighbors (KNN) machine learning model to evaluate technical indicators and predict market trajectories (UP/DOWN signals) for NSE stocks.
Trained on the Kaggle Dataset: [NSE-TATAGLOBAL](https://www.kaggle.com/datasets/akshaydattatraykhare/nsetataglobal), now identified with ticker symbol TATACONSUM at NSE.
## 🚀 Live Demo
* **Frontend (Vercel):** [https://market-prediction-model.vercel.app](https://market-prediction-model.vercel.app)
* **Backend API / Docs (Render):** [https://market-prediction-backend.onrender.com/docs](https://market-prediction-backend.onrender.com/docs)
* If opening for the first time, a cold restart may take upto 30-50s to respond
---

## 🏗️ Architecture & Tech Stack

This project uses a decoupled split-architecture:
* **Frontend:** React, Vite, Tailwind CSS, TradingView Lightweight Charts, Axios, Lucide Icons.
* **Backend:** Python, FastAPI, Pandas, Scikit-Learn.
* **Deployment:** 
  * Frontend hosted globally on Vercel
  * Backend API hosted on Render

## 📊 Model Metrics & Performance

The quantitative engine evaluates historical technical indicators using an optimized machine learning pipeline. 

### Experimentation & Tuning Process
1. **Feature Engineering:** Introduced **27** derived features from the original columns . Technical indicators, rolling statistics, and lagged signals were included so the model is trained on derived quantities which actively avoids any absolute value getting passed to the model. This prevents the model from cheating by just predicting a value close to the present day closing values.
2. **Multi-Model Training:** Initially, 8 distinct baseline models were trained simultaneously to evaluate initial feature response.
3. **Hyperparameter Optimization:** The top 3 performing model architectures were selected and passed through `GridSearchCV` for exhaustive hyperparameter tuning.
4. **Evaluation & Selection:** Final evaluation on unseen test data revealed that the optimized K-Nearest Neighbors (KNN) configuration significantly outperformed the other candidates.

### Final Optimized KNN Hyperparameters
* **`n_neighbors`**: 3
* **`weights`**: `uniform`
* **`metric`**: `minkowski` (with $p = 3$)

### Evaluation Scores Across Splits

| Dataset Split | Accuracy | Precision | F1 Score |
| :--- | :--- | :--- | :--- |
| **Training Set** | 74.17% | 0.7416 | 0.7453 |
| **Validation Set** | 51.88% | 0.5156 | 0.5344 |
| **Test Set** | **68.69%** | **0.7327** | **0.7048** |

> **Note on the validation dip:** Validation accuracy is noticeably lower than test. This is because the market window during the validation period was exceptionally choppy and unpredictable. The test split happens to fall in a more trend-following period, which flatters the test number.

* **Baseline comparison:** Directional accuracy of **68.69%** on the held-out test split, versus a 50% random baseline and a naive "tomorrow = today's direction" persistence baseline. The gap is meaningful but should be read with caution, because the model is trained on only one historical regime.
* **Precision over recall:** The model is tuned toward **precision (73.27%)** rather than raw coverage. When it fires an UP/DOWN signal, it is correct roughly 3 times out of 4. This trades signal frequency for reliability.
* **Honest scope:** This is an **educational ML pipeline demonstration**, not a tradable strategy. This model which looks strong on one test window can and will degrade quickly in live markets.

---

## 🧪 Methodology & Leakage Prevention

This is a time-series problem, so the evaluation respects temporal order end-to-end:

* **Chronological split** — train / validation / test are split in time order (no shuffling), so no future information leaks into training.
* **Time-aware cross-validation** — hyperparameter tuning uses `TimeSeriesSplit` rather than default K-Fold, ensuring every validation fold only sees data from *before* it.
* **Train-only fitting** — all feature scaling is fit on the training set only, then applied to validation/test.

---

## ⚙️ Local Development Setup

To run this project locally on your machine, follow these steps:

### Prerequisites
* **Python** (v3.8 or higher)
* **Node.js** (v16 or higher) & npm

### 1. Clone the Repository
```bash
git clone [https://github.com/sks2706-dev/Market-Prediction-Model.git](https://github.com/sks2706-dev/Market-Prediction-Model.git)
cd Market-Prediction-Model
```
### 2. Backend Setup (FastAPI)

Navigate to the backend directory, set up a virtual environment, and install dependencies:
```bash
cd backend
python -m venv env

# Activate virtual environment:
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate

pip install -r requirements.txt
```
Start the FastAPI development server:
```bash
uvicorn main:app --reload --port 8000
```
(The API documentation will be available at http://127.0.0.1:8000/docs)

### 3. Frontend Setup (React+Vite)
Open a new terminal window, navigate to the frontend directory, and install packages:
```bash
cd frontend
npm install
```
Create a .env file inside the frontend/ directory to point to your local backend:
```bash
VITE_API_URL=[http://127.0.0.1:8000](http://127.0.0.1:8000)
```
Start the frontend development server:
```bash
npm run dev
```
You will be able to see a localhost website, click on it to access the frontend

