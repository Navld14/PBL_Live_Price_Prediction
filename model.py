import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.utils import get_custom_objects
from sklearn.preprocessing import MinMaxScaler
import joblib
from data_fetch import fetch_binance_data

# Configuration
LOOKBACK = 120
FUTURE_STEPS = 24
MODEL_PATH = "lstm_model.keras"
SCALER_PATH = "scaler.pkl"

# Initialize TensorFlow with optimized settings
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Scaler handling
def get_scaler():
    if os.path.exists(SCALER_PATH):
        try:
            return joblib.load(SCALER_PATH)
        except Exception as e:
            print(f"Error loading scaler: {e}")
            return MinMaxScaler(feature_range=(0, 1))
    return MinMaxScaler(feature_range=(0, 1))

def save_scaler(scaler):
    try:
        joblib.dump(scaler, SCALER_PATH)
    except Exception as e:
        print(f"Error saving scaler: {e}")

SCALER = get_scaler()

# Data preparation
def prepare_data(df):
    # Ensure no missing or invalid values in the 'close' column
    if "close" not in df.columns or df["close"].isnull().any():
        raise ValueError("Invalid or missing 'close' column in data")

    if len(df) < LOOKBACK + FUTURE_STEPS:
        raise ValueError(f"Need at least {LOOKBACK + FUTURE_STEPS} data points")

    prices = df["close"].values.reshape(-1, 1)
    prices_scaled = SCALER.fit_transform(prices)
    save_scaler(SCALER)

    X, y = [], []
    for i in range(LOOKBACK, len(prices_scaled) - FUTURE_STEPS):
        X.append(prices_scaled[i - LOOKBACK:i])
        y.append(prices_scaled[i:i + FUTURE_STEPS])

    return np.array(X), np.array(y)

def weighted_loss(y_true, y_pred):
    time_weight = tf.exp(-tf.range(tf.shape(y_true)[0], dtype='float32') / 50.0)  # Adjust decay rate
    return tf.reduce_mean(time_weight * tf.square(y_true - y_pred))

def build_small_lstm():
    """Build a smaller LSTM model for short-term adaptation."""
    inputs = Input(shape=(LOOKBACK, 1))
    x = LSTM(32, return_sequences=True)(inputs)
    x = LSTM(16)(x)
    outputs = Dense(FUTURE_STEPS)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="mse")
    return model

def fine_tune_model():
    print("Fine-tuning existing model...")

    try:
        # ✅ (1) Use More Recent Data (Last 14 Days Instead of 7 Days)
        df_14d = fetch_binance_data(limit=4032, interval="5m")  # 14 days of 5-minute data
        df_24h = fetch_binance_data(limit=2016, interval="5m")  # Last 24 hours

        print(f"Fetched {len(df_14d)} data points for fine-tuning.")
        print(f"Fetched {len(df_24h)} data points for short-term adaptation.")

        if len(df_14d) < LOOKBACK + FUTURE_STEPS or len(df_24h) < LOOKBACK + FUTURE_STEPS:
            print("Not enough data to fine-tune.")
            raise ValueError("Insufficient data for fine-tuning")

        X_14d, y_14d = prepare_data(df_14d)
        y_14d = np.array(y_14d).reshape(-1, FUTURE_STEPS)  # Ensure correct shape for multi-step prediction

        X_24h, y_24h = prepare_data(df_24h)
        y_24h = np.array(y_24h).reshape(-1, FUTURE_STEPS)

        print(f"Fine-tuning Data Shape (14 days): {X_14d.shape}, Labels Shape: {y_14d.shape}")
        print(f"Short-term Model Data Shape (24 hours): {X_24h.shape}, Labels Shape: {y_24h.shape}")

        if len(X_14d) == 0 or len(y_14d) == 0 or len(X_24h) == 0 or len(y_24h) == 0:
            print("ERROR: Fine-tuning data is empty.")
            return None

        model_path = "lstm_model.keras"
        if not os.path.exists(model_path):
            print("ERROR: Pre-trained model not found.")
            return None

        get_custom_objects().update({"weighted_loss": weighted_loss})

        try:
            model = load_model(model_path, compile=False)
            # Ensure compatibility by checking for batch_shape issues
            if hasattr(model, "input_shape") and model.input_shape[1:] != (LOOKBACK, 1):
                print(f"Incompatible model input shape: {model.input_shape}. Rebuilding model...")
                model = build_small_lstm()  # Rebuild the model if input shape is incompatible
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mse")
        except Exception as e:
            print(f"Model loading failed due to incompatible arguments or batch_shape: {e}")
            return None, None
        print("Model loaded successfully.")

        # ✅ (4) Adjust Output Layer If Needed for FUTURE_STEPS Prediction
        # if model.output_shape[-1] != FUTURE_STEPS:
        #     print("Fixing model output shape...")
        #     model.pop()  # Remove last layer
        #     model.add(tf.keras.layers.Dense(FUTURE_STEPS))  # Add correct output layer

        # ✅ (2) Unfreeze Only the Last 3 LSTM Layers for Adaptation
        for layer in model.layers[:-3]:  
            layer.trainable = False  # Freeze old layers

        # ✅ (3) Use Lower Learning Rate for Smooth Adaptation
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mse")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]

        # ✅ (5) Apply Fine-Tuning Only When Necessary (Avoid Overfitting)
        model.fit(
            X_14d, y_14d,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        model.save(model_path)
        print("Fine-tuning complete and model saved.")

        # ✅ (6) Online Learning: Train a Small LSTM Model on Last 24 Hours
        short_term_model = build_small_lstm()
        short_term_model.fit(X_24h, y_24h, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

        return model, short_term_model

    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        return None, None

def blend_predictions(long_pred, short_pred, weight_long=0.7, weight_short=0.3):
    """Blend long-term and short-term predictions using weighted averaging."""
    # Ensure predictions are not empty and have compatible shapes
    if long_pred is None or short_pred is None or len(long_pred) == 0 or len(short_pred) == 0:
        raise ValueError("Predictions cannot be empty")

    long_pred = np.array(long_pred)
    short_pred = np.array(short_pred)
    
    if long_pred.shape != short_pred.shape:
        print("Warning: Shape mismatch in predictions. Adjusting sizes if necessary.")
        min_len = min(len(long_pred), len(short_pred))
        long_pred, short_pred = long_pred[:min_len], short_pred[:min_len]

    blended = (weight_long * long_pred) + (weight_short * short_pred)
    return blended

def inverse_transform_predictions(predictions_scaled):
    """Inverse transform scaled predictions to original scale."""
    # Ensure predictions_scaled is not empty
    if predictions_scaled is None or len(predictions_scaled) == 0:
        raise ValueError("Predictions cannot be empty for inverse transform")

    try:
        predictions_scaled = np.array(predictions_scaled)

        # If the array is 3D, reshape it to 2D
        if predictions_scaled.ndim == 3:
            predictions_scaled = predictions_scaled.reshape(-1, 1)

        # If the array is 1D, reshape it to 2D
        elif predictions_scaled.ndim == 1:
            predictions_scaled = predictions_scaled.reshape(-1, 1)

        predictions = SCALER.inverse_transform(predictions_scaled)
        return predictions.reshape(-1)  # Convert back to 1D

    except Exception as e:
        print(f"Error during inverse transform: {e}")
        return np.array([])

def update_dashboard(predictions):
    """Update the dashboard with predictions, handling invalid data."""
    try:
        if predictions is None or len(predictions) == 0:
            raise ValueError("No predictions available for dashboard update")

        # Ensure predictions are numeric
        predictions = [float(p) for p in predictions if p is not None and str(p).replace('.', '', 1).isdigit()]

        if len(predictions) == 0:
            raise ValueError("No valid numeric predictions available")

        # Update the dashboard (placeholder logic)
        print(f"Dashboard updated with predictions: {predictions[:5]}...")

    except ValueError as e:
        print(f"Dashboard update failed: {e}")
    except Exception as e:
        print(f"Unexpected error during dashboard update: {e}")

