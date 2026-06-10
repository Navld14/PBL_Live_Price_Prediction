import numpy as np
import pandas as pd
from model import fine_tune_model, SCALER, LOOKBACK, FUTURE_STEPS, inverse_transform_predictions, blend_predictions

def smooth_predictions(preds, method="ema", window=5):
    import pandas as pd
    import numpy as np
    from scipy.signal import savgol_filter

    if method == "ema":
        return pd.Series(preds).ewm(span=window, adjust=False).mean().values

    elif method == "sma":
        return pd.Series(preds).rolling(window=window).mean().fillna(method='bfill').values

    elif method == "savgol":
        if len(preds) < window:
            return preds  # Can't apply
        return savgol_filter(preds, window_length=window if window % 2 == 1 else window + 1, polyorder=2)

    elif method == "none":
        return preds

    else:
        print(f"Unknown smoothing method: {method}. Returning original.")
        return preds

def predict_next_prices(df, smoothing=True, smoothing_type="ema", window=5):
    """Predict future prices using the fine-tuned LSTM model and short-term model, with optional smoothing."""

    # Load both models
    model, short_term_model = fine_tune_model()

    if model is None or short_term_model is None:
        print("ERROR: Model not loaded, cannot predict.")
        return [], []

    try:
        # Ensure timestamps are datetime objects
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Prepare the latest data
        latest_data = df["close"].values[-LOOKBACK:].reshape(-1, 1)
        print(f"Latest data length: {len(latest_data)}")

        latest_scaled = SCALER.transform(latest_data)
        latest_scaled = latest_scaled.reshape(1, LOOKBACK, 1)

        # Make predictions using both models
        long_term_pred_scaled = model.predict(latest_scaled, verbose=0)[0]
        short_term_pred_scaled = short_term_model.predict(latest_scaled, verbose=0)[0]

        if long_term_pred_scaled.ndim == 1 and short_term_pred_scaled.ndim == 1:
            # Inverse scale predictions
            long_term_pred = inverse_transform_predictions(long_term_pred_scaled)
            short_term_pred = inverse_transform_predictions(short_term_pred_scaled)

            # Blend predictions (70% long-term, 30% short-term)
            final_predictions = blend_predictions(long_term_pred, short_term_pred, weight_long=0.9, weight_short=0.1)

            # --- Smoothing ---
            if smoothing:
                final_predictions = smooth_predictions(final_predictions, method=smoothing_type, window=window)

            # Debug: Print a sample of predictions
            print(f"Sample predictions: {final_predictions[:5]}")

            # Generate future timestamps
            last_timestamp = df["timestamp"].iloc[-1]
            print(f"Last timestamp in dataset: {last_timestamp}")

            future_timestamps = [last_timestamp + pd.Timedelta(minutes=i) for i in range(0, FUTURE_STEPS)]

            return final_predictions, future_timestamps
        else:
            print("Prediction output has unexpected shape.")

        return [], []

    except Exception as e:
        print(f"Prediction error: {e}")
        return [], []


