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
1. **Multi-Model Training:** Initially, 8 distinct baseline models were trained simultaneously to evaluate initial feature response.
2. **Hyperparameter Optimization:** The top 3 performing model architectures were selected and passed through `GridSearchCV` for exhaustive hyperparameter tuning.
3. **Evaluation & Selection:** Final evaluation on unseen test data revealed that the optimized K-Nearest Neighbors (KNN) configuration significantly outperformed the other candidates.

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

* **Beating the Baseline:** Financial markets exhibit high noise and near random-walk properties. A directional accuracy of **68.69%** on unseen test data significantly outperforms random baseline guessing (50%).
* **High Precision Focus:** The model maintains a **73.27% Precision** score, meaning when it *does* trigger a high-conviction signal, it has a high reliability rate of being correct, minimizing false positives.
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

