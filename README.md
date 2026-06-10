# PBL_1-Live_price_prediction
# 📊 Bitcoin Live Price Prediction Dashboard

A real-time Bitcoin price tracking and prediction dashboard built with an LSTM deep learning model, featuring live candlestick charts, technical indicators, and a paper trading simulation engine.

---

## 🚀 Features

- **Live Price Fetching** — Pulls real-time OHLCV data from the Binance API
- **LSTM Price Prediction** — Forecasts future BTC prices using a trained Keras LSTM model
- **Interactive Dashboard** — Built with Plotly Dash and Bootstrap (dark theme)
  - Candlestick chart
  - Price trend + prediction overlay
  - Historical price chart
- **Technical Indicators** — RSI and EMA computed in real-time using the `ta` library
- **Paper Trading Simulation** — Simulated BUY/SELL/HOLD signals with stop-loss and take-profit logic
- **Auto-refresh** — Dashboard updates every 5 minutes

---

## 🗂️ Project Structure

```
PBL_Live_Price_Prediction/
│
├── app.py              # Main Dash application — dashboard layout & callbacks
├── data_fetch.py       # Fetches live OHLCV data from Binance API
├── model.py            # LSTM model definition and training config
├── prediction.py       # Generates price predictions using the trained model
├── utils.py            # Helper functions (e.g., timestamp formatting)
├── k.py                # Additional utilities / experiments
├── lstm_model.keras    # Pre-trained LSTM model weights
└── scaler.pkl          # Fitted MinMaxScaler for data normalization
```

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Dashboard | Plotly Dash + Dash Bootstrap Components |
| Deep Learning | TensorFlow / Keras (LSTM) |
| Data Source | Binance API |
| Indicators | `ta` (Technical Analysis library) |
| Data Handling | Pandas, NumPy |
| ML Utilities | Scikit-learn (Scaler, MAPE) |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Navld14/PBL_Live_Price_Prediction.git
cd PBL_Live_Price_Prediction
```

### 2. Install dependencies
```bash
pip install dash dash-bootstrap-components plotly pandas numpy tensorflow scikit-learn ta requests
```

### 3. Run the app
```bash
python app.py
```

Then open your browser and go to: `http://localhost:8050`

---

## 📈 How It Works

1. **Data Fetching** — `data_fetch.py` pulls the last N candles of BTC/USDT data from Binance.
2. **Preprocessing** — Data is scaled using the saved `scaler.pkl` for normalization.
3. **Prediction** — `prediction.py` feeds the processed data into `lstm_model.keras` to generate future price forecasts.
4. **Indicators** — RSI and EMA are computed on the fly and displayed on the dashboard.
5. **Trading Signals** — Based on predicted vs. current price, the system generates BUY / SELL / HOLD signals with stop-loss and take-profit thresholds.
6. **Paper Trading** — A simulated portfolio tracks balance and BTC holdings based on signals (no real money involved).

---

## 📊 Model Details

- **Architecture:** LSTM (Long Short-Term Memory) neural network
- **Input:** Sequence of historical closing prices (normalized)
- **Output:** Multi-step future price predictions (`FUTURE_STEPS` defined in `model.py`)
- **Saved as:** `lstm_model.keras` + `scaler.pkl`

---

## 🧪 Evaluation Metrics

- **MAPE** (Mean Absolute Percentage Error) — measures prediction accuracy
- **Directional Accuracy** — measures how often the model correctly predicts price direction

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. The paper trading simulation does not involve real funds. Do not use this for actual financial decisions.

---

## 👨‍💻 Author

**Naveen Ladha** — B.Tech AI & ML, Symbiosis Institute of Technology, Pune  
GitHub: [@Navld14](https://github.com/Navld14)
