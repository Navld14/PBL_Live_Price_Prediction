import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from data_fetch import fetch_binance_data
from prediction import predict_next_prices
from model import FUTURE_STEPS
from utils import format_timestamp
from sklearn.metrics import mean_absolute_percentage_error
import ta  # Import ta here
import numpy as np

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# --- Paper Trading Simulation ---
INITIAL_CAPITAL = 10000
TRADE_SIZE = 100  # Amount of BTC to trade (in USD)

# --- Global Variables ---
paper_trading_balance = INITIAL_CAPITAL
btc_held = 1000
last_trade = None  # Store the last trade signal
last_df = None  # Store the last fetched dataframe

# --- Metrics History ---
r2_history = []
rmse_history = []


# --- UI Components ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("📊 Bitcoin Live Price Tracker", className="text-center text-white"))),
    dbc.Row([
        dbc.Col(dcc.Graph(id="candlestick-chart"), width=6),
        dbc.Col(dcc.Graph(id="line-chart"), width=6)
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id="historical-chart"), width=12)]),
    dbc.Row([
        dbc.Col([
            html.H4("Current Price:", className="text-white"),
            html.Div(id="current-price", className="text-white"),
        ], width=3),
        dbc.Col([
            html.H4("Last Updated:", className="text-white"),
            html.Div(id="last-updated-time", className="text-white"),
        ], width=3),
        dbc.Col([
            html.H4("Predicted Price:", className="text-white"),
            html.Div(id="predicted-price", className="text-white"),
        ], width=3),
        dbc.Col([
            html.H4("RSI:", className="text-white"),
            html.Div(id="rsi-value", className="text-white"),
        ], width=3),
        dbc.Col([
            html.H4("EMA:", className="text-white"),
            html.Div(id="ema-value", className="text-white"),
        ], width=3),
        # The following block that displayed MAPE and DA metrics has been removed:
        # html.Div([
        #     html.Div([
        #         html.H4("📈 MAPE (%)", style={"color": "orange"}),
        #         html.H2(id="mape-output")
        #     ], style={"border": "2px solid #FFD580", "padding": "10px", "margin": "10px", "borderRadius": "10px", "width": "45%"}),
        #
        #     html.Div([
        #         html.H4("📉 Directional Accuracy (%)", style={"color": "green"}),
        #         html.H2(id="da-output")
        #     ], style={"border": "2px solid #90EE90", "padding": "10px", "margin": "10px", "borderRadius": "10px", "width": "45%"}),
        # ], style={"display": "flex", "justifyContent": "space-around", "flexWrap": "wrap"})
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Trading Signal:", className="text-white"),
            html.Div(id="trading-signal", className="text-white"),
        ], width=3),
        dbc.Col([
            html.H4("Paper Trading Balance:", className="text-white"),
            html.Div(id="paper-trading-balance", className="text-white"),
        ], width=3),
        dbc.Col([
            html.H4("BTC Held:", className="text-white"),
            html.Div(id="btc-held", className="text-white"),
        ], width=3),
    ]),
    dcc.Interval(id="interval-component", interval=60000*5, n_intervals=0),
    # Hidden div to store/pass trading state
    dcc.Store(id='trading-state', data={'position': 'none', 'entry_price': 0})
], fluid=True)

def format_value(value):
    try:
        return f"${float(value):.2f}"
    except (ValueError, TypeError):
        return "N/A 1"

# --- Callbacks ---
@app.callback(
    [Output("candlestick-chart", "figure"),
     Output("line-chart", "figure"),
     Output("historical-chart", "figure"),
     Output("current-price", "children"),
     Output("last-updated-time", "children"),
     Output("predicted-price", "children"),
     Output("trading-signal", "children"),
     Output("paper-trading-balance", "children"),
     Output("btc-held", "children"),
     Output("rsi-value", "children"),
     Output("ema-value", "children"),
     Output('trading-state', 'data'),
     # Removed Output('mape-output', 'children'),
     # Removed Output('da-output', 'children')
    ],
    Input("interval-component", "n_intervals"),
    State('trading-state', 'data')
)
def update_dashboard(n, trading_state):

    global paper_trading_balance, btc_held, last_trade, last_df

    trading_state = {'position': 'none', 'entry_price': 0}

    try:
        # Fetch new data
        n=160
        df = fetch_binance_data(limit=n)
        df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['EMA'] = ta.trend.EMAIndicator(df['close']).ema_indicator()
        last_df = df.copy()  # Store the last fetched dataframe
        if df.empty:
            print("No data received from Binance API.")
            return (go.Figure(), go.Figure(), go.Figure(), "N/A 2", "N/A 3", "N/A 4",
                     last_trade, f"${paper_trading_balance:.2f}", f"{btc_held:.4f}", "N/A 5", "N/A 6", trading_state)

        current_price = df["close"].iloc[-1]
        last_updated_time = format_timestamp(df["timestamp"].iloc[-1])

        df = df.reset_index(drop=True)
        first_half = df.iloc[:n//2].copy()
        second_half = df.iloc[n//2:].copy()

        current_price = first_half["close"].iloc[-1]
        last_updated_time = format_timestamp(first_half["timestamp"].iloc[-1])

        # === Price Prediction ===
        try:
            predictions, future_timestamps = predict_next_prices(first_half)
            predicted_price = predictions[0] if len(predictions) > 0 else "N/A"

            actual_prices = second_half['close'].values

            # Match the lengths of predictions and actuals
            min_len = min(len(actual_prices), len(predictions))
            actual_prices = actual_prices[:min_len]
            predictions = predictions[:min_len]

            mape_score = mean_absolute_percentage_error(actual_prices, predictions)
            da_score = directional_accuracy(actual_prices, predictions)

        except Exception as e:
            print(f"Prediction failed: {e}")
            predicted_price = "N/A"
            mape_score = 0.0
            da_score = 0.0

        # Calculate indicators
        rsi = df["RSI"].iloc[-1] if "RSI" in df and not pd.isna(df["RSI"].iloc[-1]) else "N/A 7"
        ema = df["EMA"].iloc[-1] if "EMA" in df and not pd.isna(df["EMA"].iloc[-1]) else "N/A 8"

        # Make prediction
        try:
            predictions, future_timestamps = predict_next_prices(df)
            predicted_price = predictions[0]
            if len(predictions)==0:
                print("Pred not happening")
        except Exception as e:
            print(f"Prediction failed: {e}")
            predictions = []
            future_timestamps = []
            predicted_price = "N/A 10"

        # Trading Logic
        signal, trading_state = generate_trading_signal(current_price, float(predicted_price) if predicted_price != "N/A 11" else None, paper_trading_balance, btc_held, trading_state)
        if signal != last_trade:
            print(f"New Trading Signal: {signal}")
            last_trade = signal

        paper_trading_balance, btc_held = execute_trade(signal, current_price, paper_trading_balance, btc_held)

        # Candlestick Chart
        candlestick_fig = go.Figure(data=[go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"]
        )])
        candlestick_fig.update_layout(template="plotly_dark", title="Bitcoin Candlestick Chart")

        # Price Trend & Prediction
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["close"],
            mode="lines",
            name="BTC Price"
        ))
        if len(predictions) > 0 and len(future_timestamps) > 0:
            line_fig.add_trace(go.Scatter(
                x=future_timestamps,
                y=predictions,
                mode="lines",
                name="Predicted Price",
                line=dict(color="red", dash="dot")
            ))
        line_fig.update_layout(template="plotly_dark", title="Bitcoin Price Trend & Prediction")

        # Historical Chart
        historical_fig = go.Figure()
        historical_fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["close"],
            mode="lines",
            name="Historical BTC Prices"
        ))
        if len(predictions) > 0 and len(future_timestamps) > 0:
            historical_fig.add_trace(go.Scatter(
                x=future_timestamps,
                y=predictions,
                mode="lines",
                name="Predicted Price",
                line=dict(color="red", dash="dot")
            ))
        historical_fig.update_layout(template="plotly_dark", title="Historical Bitcoin Prices with Prediction")

        return (candlestick_fig, line_fig, historical_fig, format_value(current_price), last_updated_time,
                format_value(predicted_price), signal, format_value(paper_trading_balance), format_value(btc_held),
                str(rsi), str(ema), trading_state) # Removed MAPE and DA return values

    except Exception as e:
        print(f"Dashboard update failed: {e}")
        return (go.Figure(), go.Figure(), go.Figure(), "N/A 12", "N/A 13", "N/A 14",
                last_trade, format_value(paper_trading_balance), format_value(btc_held), "N/A 15", "N/A 16", trading_state) # Removed MAPE and DA return values

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    min_len = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    return mape

def directional_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    min_len = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    correct_directions = np.sum(np.sign(true_diff) == np.sign(pred_diff))
    da = (correct_directions / len(true_diff)) * 100
    return da

# --- Trading Logic Functions ---
def generate_trading_signal(current_price, predicted_price, balance, btc_held, trading_state,
                             buy_threshold=0.005, sell_threshold=0.05,
                             stop_loss_percent=0.005, take_profit_percent=0.01):
    """Generates a trading signal based on predicted price movement."""
    signal = "HOLD"
    position = trading_state['position']
    entry_price = trading_state['entry_price']

    if predicted_price is None:
        return "HOLD", trading_state

    # Calculate price change percentage
    price_change_percent = (predicted_price - current_price) / current_price

    # Stop Loss & Take Profit Logic (only when holding a position)
    if position == 'long':
        if current_price <= entry_price * (1 - stop_loss_percent):
            return "SELL (Stop Loss)", {'position': 'none', 'entry_price': 0}
        elif current_price >= entry_price * (1 + take_profit_percent):
            return "SELL (Take Profit)", {'position': 'none', 'entry_price': 0}
    elif position == 'short':
        if current_price >= entry_price * (1 + stop_loss_percent):
            return "SELL (Stop Loss)", {'position': 'none', 'entry_price': 0}
        elif current_price <= entry_price * (1 - take_profit_percent):
            return "BUY (Take Profit)", {'position': 'none', 'entry_price': 0}

    # Entry conditions (Only when no position is open)
    if position == 'none':
        if price_change_percent > buy_threshold:
            return "BUY", {'position': 'long', 'entry_price': current_price}
        elif price_change_percent < sell_threshold:
            return "SELL", {'position': 'short', 'entry_price': current_price}

    return signal, trading_state


def execute_trade(signal, current_price, balance, btc_held, sell_percentage=0.5):
    """Executes a trade based on the signal and updates the paper trading balance."""
    global TRADE_SIZE

    if signal == "BUY":
        if balance >= TRADE_SIZE:
            btc_to_buy = TRADE_SIZE / current_price
            balance -= TRADE_SIZE
            btc_held += btc_to_buy
            print(f"Bought {btc_to_buy:.4f} BTC at {current_price:.2f}")

    elif signal.startswith("SELL"):  # Handle both SELL and Stop Loss
        # Dynamic selling strategy based on profit/loss conditions
        if btc_held > 0:
            # Define selling logic
            if signal == "SELL (Take Profit)":
                btc_to_sell = btc_held * 0.75  # Take profit by selling 75% of holdings
            elif signal == "SELL (Stop Loss)":
                btc_to_sell = btc_held * 0.5  # Cut losses by selling 50% of holdings
            else:
                btc_to_sell = btc_held * sell_percentage  # Default sell percentage

            # Ensure at least a minimum amount is sold
            btc_to_sell = max(btc_to_sell, btc_held * 0.1)  # Sell at least 10%
            btc_to_sell = min(btc_to_sell, btc_held)  # Do not oversell

            balance += btc_to_sell * current_price
            btc_held -= btc_to_sell
            print(f"Sold {btc_to_sell:.4f} BTC at {current_price:.2f}")

    return balance, btc_held



if __name__ == "__main__":
    app.run(debug=True, port=8050)