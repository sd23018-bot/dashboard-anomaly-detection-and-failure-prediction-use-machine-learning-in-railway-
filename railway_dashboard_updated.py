"""
Railway Critical Component — Anomaly Detection & Failure Prediction Dashboard
==============================================================================
Built to accompany: "Anomaly Detection and Failure Prediction Using Machine
Learning in Railway for Critical Component"

Run with:
    streamlit run app.py
"""

import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    IsolationForest,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.neighbors import LocalOutlierFactor, KNeighborsRegressor
from sklearn.svm import OneClassSVM, SVR
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ----- Optional dependencies for time‑series & deep learning -----
try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ----------------------------------------------------------------------
# ANOMALY MODELS (all with fixed default parameters)
# ----------------------------------------------------------------------
ANOMALY_MODELS = [
    "Isolation Forest",
    "Local Outlier Factor",
    "One-Class SVM",
]

# Default parameters for anomaly models (used internally)
ANOMALY_DEFAULTS = {
    "Isolation Forest": {"n_estimators": 200, "contamination": 0.04, "max_features": 0.7},
    "Local Outlier Factor": {"n_neighbors": 20, "contamination": 0.04},
    "One-Class SVM": {"nu": 0.04},
    "Elliptic Envelope (Robust Covariance)": {"contamination": 0.04, "support_fraction": 0.8},
    "Gaussian Mixture Model": {"n_components": 3, "contamination": 0.04},
    "PCA Reconstruction Error": {"n_components": 3, "contamination": 0.04},
    "Autoencoder (Deep Learning)": {"seq_len": 10, "units": 32, "epochs": 50, "batch_size": 32, "contamination": 0.04},
}

# ----------------------------------------------------------------------
# FAILURE PREDICTION MODELS (all with fixed default parameters)
# ----------------------------------------------------------------------
REGRESSION_MODELS = [
    "Random Forest",
    "Gradient Boosting",
    "Support Vector Regression",
]

# Default parameters for the three selected regression models
REGRESSION_DEFAULTS = {
    "Random Forest": {"n_estimators": 200, "max_depth": 20},
    "Gradient Boosting": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
    "Support Vector Regression": {"C": 2.0, "epsilon": 0.01},
}

st.set_page_config(
    page_title="Railway Anomaly Detection & Failure Prediction",
    page_icon="🚆",
    layout="wide",
)

# --------------------------------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------------------------------
for key in ["df_raw", "df", "feature_columns", "train", "test", "scaler",
            "det_result", "det_threshold", "det_model_name",
            "pred_result", "native_threshold", "y_scaler",
            "forecast_df", "forecast_sensor_name", "forecast_thresholds",
            "anomaly_comparison", "prediction_comparison"]:
    if key not in st.session_state:
        st.session_state[key] = None


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def load_excel(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    if len(xls.sheet_names) > 1:
        sheet = st.sidebar.selectbox("Select sheet", xls.sheet_names)
    else:
        sheet = xls.sheet_names[0]
    return pd.read_excel(xls, sheet_name=sheet)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=True, sheet_name="results")
    return buf.getvalue()


def apply_threshold(scores: pd.Series, method: str, native_thr: float = None, **kwargs) -> tuple[pd.Series, float]:
    if method == "Auto (model's native threshold)":
        if native_thr is None:
            raise ValueError("Native threshold not available for this model.")
        thr = native_thr
        flag = (scores <= thr).astype(int)
    elif method == "Percentile (worst X%)":
        pct = kwargs["percentile"]
        thr = np.percentile(scores, pct)
        flag = (scores <= thr).astype(int)
    elif method == "Mean - K*Std":
        k = kwargs["k"]
        thr = scores.mean() - k * scores.std()
        flag = (scores <= thr).astype(int)
    elif method == "Manual value":
        thr = kwargs["manual_value"]
        flag = (scores <= thr).astype(int)
    else:
        raise ValueError("Unknown threshold method")
    return flag, thr


def compute_asoi_detailed(scores, percentile=5):
    """
    Compute ASOI based on the formula:
    ASOI = |mu_normal - mu_anomaly| / (std_normal + std_anomaly)
    Returns: (asoi, raw_gap, mu_norm, mu_anom, std_norm, std_anom)
    """
    scores = np.asarray(scores).flatten()
    if len(scores) < 10:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Split into anomaly and normal groups (lowest percentile% = anomalies)
    threshold = np.percentile(scores, percentile)
    anom_scores = scores[scores <= threshold]
    norm_scores = scores[scores > threshold]

    if len(anom_scores) < 2 or len(norm_scores) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    mu_norm = np.mean(norm_scores)
    mu_anom = np.mean(anom_scores)
    std_norm = np.std(norm_scores, ddof=1) if len(norm_scores) > 1 else 0
    std_anom = np.std(anom_scores, ddof=1) if len(anom_scores) > 1 else 0

    raw_gap = mu_norm - mu_anom

    denominator = std_norm + std_anom
    if denominator == 0:
        asoi = 0.0
    else:
        asoi = abs(mu_norm - mu_anom) / denominator

    return asoi, raw_gap, mu_norm, mu_anom, std_norm, std_anom


def fit_anomaly_model(model_name, X_train, X_test):
    """Fit model with its default parameters and return scores + native threshold."""
    params = ANOMALY_DEFAULTS[model_name].copy()
    native_threshold = None

    if model_name == "Isolation Forest":
        model = IsolationForest(
            n_estimators=params["n_estimators"],
            contamination=params["contamination"],
            max_features=params["max_features"],
            random_state=42,
        )
        model.fit(X_train)
        train_scores = model.score_samples(X_train)
        test_scores = model.score_samples(X_test)
        native_threshold = model.offset_

    elif model_name == "Local Outlier Factor":
        model = LocalOutlierFactor(
            n_neighbors=params["n_neighbors"],
            contamination=params["contamination"],
            novelty=True,
        )
        model.fit(X_train)
        train_scores = model.score_samples(X_train)
        test_scores = model.score_samples(X_test)
        native_threshold = model.offset_

    elif model_name == "One-Class SVM":
        model = OneClassSVM(kernel="rbf", nu=params["nu"], gamma="scale")
        model.fit(X_train)
        train_scores = model.decision_function(X_train)
        test_scores = model.decision_function(X_test)
        native_threshold = 0.0

    elif model_name == "Elliptic Envelope (Robust Covariance)":
        model = EllipticEnvelope(
            contamination=params["contamination"],
            support_fraction=params["support_fraction"],
            random_state=42,
        )
        model.fit(X_train)
        train_scores = model.decision_function(X_train)
        test_scores = model.decision_function(X_test)
        native_threshold = 0.0

    elif model_name == "Gaussian Mixture Model":
        model = GaussianMixture(
            n_components=params["n_components"],
            covariance_type="full",
            random_state=42,
        )
        model.fit(X_train)
        train_scores = model.score_samples(X_train)
        test_scores = model.score_samples(X_test)
        native_threshold = np.percentile(train_scores, params["contamination"] * 100)

    elif model_name == "PCA Reconstruction Error":
        n_comp = min(params["n_components"], X_train.shape[1])
        model = PCA(n_components=n_comp, random_state=42)
        model.fit(X_train)

        def _neg_recon_error(X):
            X_proj = model.transform(X)
            X_recon = model.inverse_transform(X_proj)
            err = np.mean((np.asarray(X) - X_recon) ** 2, axis=1)
            return -err

        train_scores = _neg_recon_error(X_train)
        test_scores = _neg_recon_error(X_test)
        native_threshold = np.percentile(train_scores, params["contamination"] * 100)

    elif model_name == "Autoencoder (Deep Learning)":
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Please run: pip install tensorflow")
        seq_len = params["seq_len"]
        units = params["units"]
        epochs = params["epochs"]
        batch_size = params["batch_size"]
        n_features = X_train.shape[1]

        def create_sequences(data, seq_len):
            X_seq = []
            for i in range(len(data) - seq_len + 1):
                X_seq.append(data[i:i+seq_len])
            return np.array(X_seq)

        X_train_seq = create_sequences(X_train.values, seq_len)
        model = Sequential()
        model.add(LSTM(units, activation='relu', input_shape=(seq_len, n_features)))
        model.add(RepeatVector(seq_len))
        model.add(LSTM(units, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(n_features)))
        model.compile(optimizer='adam', loss='mse')
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0)
        model.fit(X_train_seq, X_train_seq,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=[early_stop],
                  verbose=0)

        def reconstruction_error(data):
            seq_data = create_sequences(data, seq_len)
            reconst = model.predict(seq_data, verbose=0)
            mse = np.mean((seq_data - reconst) ** 2, axis=(1, 2))
            full_errors = np.full(len(data), mse[0])
            full_errors[seq_len-1:] = mse
            return full_errors

        train_errors = reconstruction_error(X_train.values)
        test_errors = reconstruction_error(X_test.values)
        train_scores = -train_errors
        test_scores = -test_errors
        native_threshold = np.percentile(train_scores, params["contamination"] * 100)

    else:
        raise ValueError(f"Unknown anomaly model: {model_name}")

    return model, train_scores, test_scores, native_threshold


def fit_regression_model(model_name, X_train, y_train, X_test, y_test):
    """Fit regression model with defaults and return predictions and metrics."""
    params = REGRESSION_DEFAULTS[model_name].copy()

    if model_name in ["Random Forest", "Extra Trees", "XGBoost", "Gradient Boosting",
                      "Support Vector Regression", "Ridge Regression", "K-Nearest Neighbors"]:
        # Classic sklearn models
        if model_name == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42,
                n_jobs=-1,
            )
        elif model_name == "Extra Trees":
            model = ExtraTreesRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42,
                n_jobs=-1,
            )
        elif model_name == "XGBoost":
            model = xgb.XGBRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                random_state=42,
                n_jobs=-1,
            )
        elif model_name == "Gradient Boosting":
            model = GradientBoostingRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                random_state=42,
            )
        elif model_name == "Support Vector Regression":
            model = SVR(kernel="rbf", C=params["C"], epsilon=params["epsilon"], gamma="scale")
        elif model_name == "Ridge Regression":
            model = Ridge(alpha=params["alpha"], random_state=42)
        elif model_name == "K-Nearest Neighbors":
            model = KNeighborsRegressor(n_neighbors=params["n_neighbors"], weights=params["weights"])
        else:
            raise ValueError("Unknown classic regressor")

        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        return model, pred_train, pred_test

    elif model_name == "ARIMA (Time Series)":
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is not installed")
        model = ARIMA(y_train, order=(params["p"], params["d"], params["q"]))
        fitted = model.fit()
        pred_train = fitted.predict(start=0, end=len(y_train)-1, dynamic=False)
        pred_test = fitted.forecast(steps=len(y_test))
        return fitted, pred_train, pred_test

    elif model_name == "LSTM (Deep Learning)":
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        lookback = params["lookback"]
        units = params["units"]
        epochs = params["epochs"]
        batch_size = params["batch_size"]
        n_features = X_train.shape[1]

        # Scale target
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

        def create_seq(X, y, lookback):
            Xs, ys = [], []
            for i in range(lookback, len(X)):
                Xs.append(X[i-lookback:i])
                ys.append(y[i])
            return np.array(Xs), np.array(ys)

        X_train_seq, y_train_seq = create_seq(X_train.values, y_train_scaled, lookback)
        X_test_seq, y_test_seq = create_seq(X_test.values, y_test_scaled, lookback)

        model = Sequential()
        model.add(LSTM(units, activation='relu', input_shape=(lookback, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0)
        model.fit(X_train_seq, y_train_seq,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=[early_stop],
                  verbose=0)

        pred_train_scaled = model.predict(X_train_seq, verbose=0).flatten()
        pred_test_scaled = model.predict(X_test_seq, verbose=0).flatten()

        pred_train = y_scaler.inverse_transform(pred_train_scaled.reshape(-1, 1)).flatten()
        pred_test = y_scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).flatten()

        # Align with indices (first `lookback` points have no prediction)
        full_train = np.full(len(y_train), np.nan)
        full_train[lookback:] = pred_train
        full_test = np.full(len(y_test), np.nan)
        full_test[lookback:] = pred_test

        return model, full_train, full_test

    else:
        raise ValueError(f"Unknown regression model: {model_name}")


def classify_anomaly(sensor_col, val, prev_val, window_vals, spike_thr, slope_thr):
    delta = val - prev_val
    slope = np.mean(np.diff(window_vals)) if len(window_vals) > 1 else 0
    consec_neg = sum(1 for d in np.diff(window_vals) if d < -0.1 * spike_thr)
    consec_pos = sum(1 for d in np.diff(window_vals) if d > 0.1 * spike_thr)
    base = f"Detected by ML model because {sensor_col} behaviour is different from the learned normal pattern"
    if delta >= spike_thr:
        atype = f"{sensor_col} sudden spike up"
        reason = f"Abnormal type: {atype}; {base}; {sensor_col} jumped suddenly (change=+{delta:.2f})"
    elif delta <= -spike_thr:
        atype = f"{sensor_col} sudden drop"
        reason = f"Abnormal type: {atype}; {base}; {sensor_col} dropped suddenly (change={delta:.2f})"
    elif slope < -slope_thr and consec_neg >= 2:
        atype = f"{sensor_col} gradually decreasing"
        reason = f"Abnormal type: {atype}; {base}"
    elif slope > slope_thr and consec_pos >= 2:
        atype = f"{sensor_col} gradually increasing"
        reason = f"Abnormal type: {atype}; {base}"
    elif delta >= spike_thr or slope < -slope_thr:
        atype = f"{sensor_col} sudden drop, {sensor_col} gradually decreasing"
        reason = f"Abnormal type: {atype}; {base}; {sensor_col} dropped suddenly (change={delta:.2f})"
    else:
        atype = f"{sensor_col} unusual combination of pattern"
        reason = f"Abnormal type: {atype}; {base}"
    return atype, reason


def build_anomaly_table(det_result, sensor_col, window_size=5, spike_thr=2.5, slope_thr=0.3):
    det_result = det_result.copy()
    anom_df = det_result[det_result["anomaly"] == 1]
    all_vals = det_result[sensor_col].values
    records = []
    for ts, row in anom_df.iterrows():
        pos = det_result.index.get_loc(ts)
        val = row[sensor_col]
        prev_val = all_vals[pos - 1] if pos > 0 else val
        start = max(0, pos - window_size + 1)
        window = all_vals[start: pos + 1]
        atype, reason = classify_anomaly(sensor_col, val, prev_val, window, spike_thr, slope_thr)
        records.append({
            "Timestamp": ts.strftime("%Y-%m-%d %H:%M"),
            sensor_col: round(float(val), 2),
            "Abnormal Type": atype,
            "Reason": reason,
        })
    return pd.DataFrame(records)


def infer_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    diffs = index.to_series().diff().dropna()
    return diffs.median() if len(diffs) else pd.Timedelta(hours=1)


def build_linear_forecast(series: pd.Series, horizon: int, trend_window: int):
    tail = series.iloc[-trend_window:]
    x = np.arange(len(tail))
    slope, intercept = np.polyfit(x, tail.values, 1)
    step = infer_step(series.index)
    future_index = pd.date_range(start=series.index[-1] + step, periods=horizon, freq=step)
    future_x = np.arange(len(tail), len(tail) + horizon)
    forecast_vals = slope * future_x + intercept
    resid_std = tail.std()
    upper_band = forecast_vals + 1.96 * resid_std
    lower_band = forecast_vals - 1.96 * resid_std
    return pd.DataFrame(
        {"forecast": forecast_vals, "upper_ci": upper_band, "lower_ci": lower_band},
        index=future_index,
    )


# --------------------------------------------------------------------------
# SIDEBAR — DATA INPUT & CONFIGURATION
# --------------------------------------------------------------------------
st.sidebar.title("🚆 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload sensor dataset (Excel .xlsx)", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df_raw = load_excel(uploaded_file)
        st.session_state.df_raw = df_raw
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")

# Main title
st.title("🚆 Railway Critical Component — Anomaly Detection & Failure Prediction")
st.caption("All controls are in the sidebar. Main view shows forecast and anomaly explanation.")

if st.session_state.df_raw is None:
    st.info("👈 Upload an Excel file with sensor readings to get started.")
    st.stop()

df_raw = st.session_state.df_raw

# --------------------------------------------------------------------------
# SIDEBAR — COLUMN CONFIGURATION
# --------------------------------------------------------------------------
st.sidebar.header("📊 Data Configuration")
cols = df_raw.columns.tolist()
ts_guess = next((c for c in cols if "time" in c.lower() or "date" in c.lower()), cols[0])
ts_col = st.sidebar.selectbox("Timestamp column", cols, index=cols.index(ts_guess))

df = df_raw.copy()
try:
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()
except Exception as e:
    st.error(f"Could not parse '{ts_col}' as a timestamp: {e}")
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
feature_columns = st.sidebar.multiselect(
    "Sensor / feature columns to use", numeric_cols, default=numeric_cols
)

if not feature_columns:
    st.warning("Select at least one numeric feature column.")
    st.stop()

st.session_state.feature_columns = feature_columns
numeric_df = df[feature_columns].dropna()

# Train/Test split
st.sidebar.subheader("Train/Test Split")
split_pct = st.sidebar.slider("Training set size (%)", 50, 95, 80)

with st.sidebar.expander("ℹ️ Why adjust the Train/Test Split?", expanded=True):
    st.markdown(
        """
        **Training data** is used to teach the models the normal sensor patterns
        and relationships.

        **Testing data** is kept separate and is used to check how well the
        trained models perform on unseen readings.

        **Why change the percentage?**
        - **More training data (e.g. 80/20):** gives the model more examples to learn from.
        - **More testing data (e.g. 70/30):** gives a larger independent set for evaluation.
        - **Smaller training set:** may make the model less reliable if it has too few examples.

        **Important:** The data is split chronologically (earlier records for
        training, later records for testing), not randomly. This better reflects
        how a model would be used to predict future railway sensor behaviour.

        **Default recommendation: 80% training / 20% testing.**
        """
    )

train_size = int(len(numeric_df) * split_pct / 100)
train_raw = numeric_df.iloc[:train_size].copy()
test_raw = numeric_df.iloc[train_size:].copy()

scaler = StandardScaler().fit(train_raw[feature_columns])
train = pd.DataFrame(scaler.transform(train_raw[feature_columns]), columns=feature_columns, index=train_raw.index)
test = pd.DataFrame(scaler.transform(test_raw[feature_columns]), columns=feature_columns, index=test_raw.index)
st.session_state.train, st.session_state.test, st.session_state.scaler = train, test, scaler

st.sidebar.caption(f"Training: {split_pct}% ({train.shape[0]:,} rows)  |  Testing: {100-split_pct}% ({test.shape[0]:,} rows)")

# --------------------------------------------------------------------------
# SIDEBAR — ANOMALY DETECTION CONTROLS
# --------------------------------------------------------------------------
st.sidebar.header("🔎 Anomaly Detection")

# Model selection
det_model = st.sidebar.selectbox("Anomaly Model", ANOMALY_MODELS, key="det_model_sidebar")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Fit Manual Model"):
        with st.spinner("Fitting selected model..."):
            try:
                _, train_s, test_s, native_thr = fit_anomaly_model(det_model, train[feature_columns], test[feature_columns])
                full_scores = pd.Series(
                    np.concatenate([train_s, test_s]),
                    index=list(train.index) + list(test.index),
                    name="anomaly_score",
                ).sort_index()
                result = numeric_df.loc[full_scores.index].copy()
                result["anomaly_score"] = full_scores
                result["split"] = ["train"] * len(train_s) + ["test"] * len(test_s)
                st.session_state.det_result = result
                st.session_state.det_model_name = det_model
                st.session_state.native_threshold = native_thr
                st.sidebar.success(f"✅ {det_model} fitted.")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

with col2:
    if st.button("🚀 Auto Find Best"):
        asoi_dict = {}
        detail_dict = {}
        with st.spinner("Running all models..."):
            for m in ANOMALY_MODELS:
                try:
                    model_start = time.perf_counter()
                    _, train_s, test_s, _ = fit_anomaly_model(
                        m, train[feature_columns], test[feature_columns]
                    )
                    fit_seconds = time.perf_counter() - model_start
                    all_scores = np.concatenate([train_s, test_s])
                    asoi_val, raw_gap, mu_norm, mu_anom, std_norm, std_anom = compute_asoi_detailed(
                        all_scores, percentile=5
                    )
                    # Use the same 5th-percentile score rule for a comparable
                    # anomaly-rate indicator across the three models.
                    common_threshold = np.percentile(all_scores, 5)
                    anomaly_rate = float(np.mean(all_scores <= common_threshold) * 100)
                    asoi_dict[m] = asoi_val
                    detail_dict[m] = {
                        "ASOI": asoi_val,
                        "Anomaly Rate (%)": anomaly_rate,
                        "Fit Time (seconds)": fit_seconds,
                        "Raw Gap": raw_gap,
                        "Mu Normal": mu_norm,
                        "Mu Anomaly": mu_anom,
                        "Std Normal": std_norm,
                        "Std Anomaly": std_anom,
                    }
                except Exception as e:
                    asoi_dict[m] = -np.inf
                    detail_dict[m] = {'ASOI': -np.inf, 'Raw Gap': -np.inf, 'Mu Normal': -np.inf, 'Mu Anomaly': -np.inf, 'Std Normal': -np.inf, 'Std Anomaly': -np.inf}
                    st.sidebar.warning(f"⚠️ {m} failed: {e}")

        valid_asoi = {k: v for k, v in asoi_dict.items() if v != -np.inf}
        if not valid_asoi:
            st.sidebar.error("All models failed.")
        else:
            best_model = max(valid_asoi, key=valid_asoi.get)
            st.session_state["best_anomaly_model"] = best_model
            # Refit best model
            _, train_s, test_s, native_thr = fit_anomaly_model(best_model, train[feature_columns], test[feature_columns])
            full_scores = pd.Series(
                np.concatenate([train_s, test_s]),
                index=list(train.index) + list(test.index),
                name="anomaly_score",
            ).sort_index()
            result = numeric_df.loc[full_scores.index].copy()
            result["anomaly_score"] = full_scores
            result["split"] = ["train"] * len(train_s) + ["test"] * len(test_s)
            st.session_state.det_result = result
            st.session_state.det_model_name = best_model
            st.session_state.native_threshold = native_thr
            st.session_state.anomaly_comparison = pd.DataFrame(detail_dict).T.sort_values('ASOI', ascending=False)
            st.sidebar.success(f"✅ Best: {best_model} (ASOI = {valid_asoi[best_model]:.4f})")


# --------------------------------------------------------------------------
# SIDEBAR — THRESHOLD METHOD
# --------------------------------------------------------------------------
if st.session_state.det_result is not None:
    st.sidebar.subheader("Threshold Method")
    
    thr_method = st.selectbox(
        "Choose threshold method",
        ["Auto (model's native threshold)", "Percentile (worst X%)", "Mean - K*Std", "Manual value"],
        key="thr_method_sidebar"
    )
    
    with st.expander("ℹ️ What does each threshold method mean?"):
        st.markdown("""
        - **Auto** – uses the model's built‑in threshold (e.g., offset from Isolation Forest).
        - **Percentile** – flags the lowest X% of anomaly scores.
        - **Mean - K*Std** – flags points K standard deviations below the mean score.
        - **Manual** – you set a fixed cutoff value.
        
        **Note:** Lower anomaly score = more anomalous.
        """)
    
    thr_kwargs = {}
    if thr_method == "Percentile (worst X%)":
        thr_kwargs["percentile"] = st.sidebar.slider("Percentile", 1, 25, 5)
    elif thr_method == "Mean - K*Std":
        thr_kwargs["k"] = st.sidebar.slider("K", 0.5, 4.0, 2.0, 0.5)
    elif thr_method == "Manual value":
        scores = st.session_state.det_result["anomaly_score"]
        smin, smax = float(scores.min()), float(scores.max())
        thr_kwargs["manual_value"] = st.sidebar.slider("Threshold value", smin, smax, float(scores.quantile(0.05)))

    # Apply threshold automatically
    try:
        result = st.session_state.det_result
        native_thr = st.session_state.native_threshold if st.session_state.native_threshold is not None else None
        flag, thr_value = apply_threshold(
            result["anomaly_score"],
            thr_method,
            native_thr=native_thr,
            **thr_kwargs
        )
        result["anomaly"] = flag
        st.session_state.det_result = result
        st.session_state.det_threshold = thr_value
        st.sidebar.metric("Anomalies flagged", int(result["anomaly"].sum()))
        st.sidebar.metric("Threshold", f"{thr_value:.4f}")
    except Exception as e:
        st.sidebar.error(f"Threshold error: {e}")
else:
    st.sidebar.info("Fit an anomaly model first.")

# --------------------------------------------------------------------------
# SIDEBAR — FAILURE PREDICTION CONTROLS
# --------------------------------------------------------------------------
st.sidebar.header("📈 Failure Prediction")
if st.session_state.det_result is not None and "anomaly" in st.session_state.det_result.columns:
    reg_model = st.sidebar.selectbox("Prediction Model", REGRESSION_MODELS, key="reg_model_sidebar")
    
    pred_col1, pred_col2 = st.sidebar.columns(2)
    with pred_col1:
        if st.button("Fit Manual Prediction"):
            with st.spinner("Training selected model..."):
                try:
                    det = st.session_state.det_result
                    y_train = det.loc[train.index, "anomaly_score"]
                    y_test = det.loc[test.index, "anomaly_score"]
                    _, pred_train, pred_test = fit_regression_model(reg_model, train[feature_columns], y_train,
                                                                     test[feature_columns], y_test)
                    # Compute metrics
                    valid_mask = ~np.isnan(pred_test)
                    if np.sum(valid_mask) > 0:
                        mae = mean_absolute_error(y_test[valid_mask], pred_test[valid_mask])
                        rmse = np.sqrt(mean_squared_error(y_test[valid_mask], pred_test[valid_mask]))
                        r2 = r2_score(y_test[valid_mask], pred_test[valid_mask])
                    else:
                        mae, rmse, r2 = np.nan, np.nan, np.nan
                    # Store result
                    y_true_all = pd.concat([y_train, y_test])
                    y_pred_all = np.concatenate([pred_train, pred_test])
                    pred_df = pd.DataFrame({
                        "actual_score": y_true_all,
                        "predicted_score": y_pred_all,
                        "split": ["train"] * len(y_train) + ["test"] * len(y_test),
                    }, index=y_true_all.index).sort_index()
                    st.session_state.pred_result = pred_df
                    st.session_state["reg_model_name"] = reg_model
                    st.session_state["reg_metrics"] = {"MAE": mae, "RMSE": rmse, "R2": r2}
                    st.sidebar.success(f"✅ {reg_model} trained.  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
    
    with pred_col2:
        if st.button("🚀 Find Best Prediction"):
            det = st.session_state.det_result
            y_train = det.loc[train.index, "anomaly_score"]
            y_test = det.loc[test.index, "anomaly_score"]
            best_reg = None
            best_rmse = np.inf
            pred_metrics = {}
            with st.spinner("Running prediction models..."):
                for m in REGRESSION_MODELS:
                    try:
                        model_start = time.perf_counter()
                        _, pred_train, pred_test = fit_regression_model(
                            m, train[feature_columns], y_train,
                            test[feature_columns], y_test
                        )
                        fit_seconds = time.perf_counter() - model_start
                        valid = ~np.isnan(pred_test)
                        if np.sum(valid) > 0:
                            mae = mean_absolute_error(y_test[valid], pred_test[valid])
                            rmse = np.sqrt(mean_squared_error(y_test[valid], pred_test[valid]))
                            r2 = r2_score(y_test[valid], pred_test[valid])
                        else:
                            mae, rmse, r2 = np.nan, np.nan, np.nan
                        pred_metrics[m] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Fit Time (seconds)': fit_seconds}
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_reg = m
                            best_pred_train = pred_train
                            best_pred_test = pred_test
                    except Exception as e:
                        pred_metrics[m] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'Fit Time (seconds)': np.nan}
                        continue
            if best_reg is not None:
                # Store results for best model
                y_true_all = pd.concat([y_train, y_test])
                y_pred_all = np.concatenate([best_pred_train, best_pred_test])
                pred_df = pd.DataFrame({
                    "actual_score": y_true_all,
                    "predicted_score": y_pred_all,
                    "split": ["train"] * len(y_train) + ["test"] * len(y_test),
                }, index=y_true_all.index).sort_index()
                st.session_state.pred_result = pred_df
                st.session_state["reg_model_name"] = best_reg
                st.session_state["reg_metrics"] = pred_metrics[best_reg]
                st.session_state.prediction_comparison = pd.DataFrame(pred_metrics).T.sort_values('RMSE', ascending=True)
                st.sidebar.success(f"✅ Best: {best_reg} (RMSE = {best_rmse:.4f})")
            else:
                st.sidebar.error("All models failed.")
else:
    st.sidebar.info("Run anomaly detection first.")


# --------------------------------------------------------------------------
# SIDEBAR — EXPORT
# --------------------------------------------------------------------------
st.sidebar.header("⬇️ Export")
if st.session_state.det_result is not None:
    export_df = st.session_state.det_result.copy()
    if st.session_state.pred_result is not None:
        export_df = export_df.join(st.session_state.pred_result[["predicted_score"]], how="left")
        if "predicted_failure" in st.session_state.pred_result.columns:
            export_df = export_df.join(st.session_state.pred_result[["predicted_failure"]], how="left")
    st.sidebar.download_button(
        "Download CSV",
        data=export_df.to_csv(index=True).encode("utf-8"),
        file_name="results.csv",
        mime="text/csv",
    )
    st.sidebar.download_button(
        "Download Excel",
        data=to_excel_bytes(export_df),
        file_name="results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# --------------------------------------------------------------------------
# MAIN PAGE — DASHBOARD AND MODEL PERFORMANCE TABS
# --------------------------------------------------------------------------
dashboard_tab, performance_tab = st.tabs(["🚆 Dashboard", "📊 Model Performance"])

with dashboard_tab:
    # --------------------------------------------------------------------------
    # MAIN PAGE — FORECAST CHART
    # --------------------------------------------------------------------------
    st.header("🔮 Forecast & Failure Zone")
    if st.session_state.det_result is None or "anomaly" not in st.session_state.det_result.columns:
        st.info("Please fit an anomaly model and apply a threshold first (see sidebar).")
    else:
        det = st.session_state.det_result

        col1, col2 = st.columns(2)
        with col1:
            sensor_default = "TP2" if "TP2" in feature_columns else feature_columns[0]
            sensor_col = st.selectbox("Sensor for forecast", feature_columns, index=feature_columns.index(sensor_default), key="forecast_sensor")
        with col2:
            horizon = st.slider("Forecast horizon (steps ahead)", 5, 500, 168, key="forecast_horizon")

        if st.button("🚀 Generate Forecast"):
            normal_vals = det[det["anomaly"] == 0][sensor_col]
            upper_threshold = normal_vals.max()
            lower_threshold = normal_vals.min()
            df_forecast = build_linear_forecast(det[sensor_col], horizon, 100)
            df_forecast["anomaly"] = (
                (df_forecast["forecast"] > upper_threshold) | (df_forecast["forecast"] < lower_threshold)
            ).astype(int)
            st.session_state["forecast_df"] = df_forecast
            st.session_state["forecast_sensor_name"] = sensor_col
            st.session_state["forecast_thresholds"] = (upper_threshold, lower_threshold)

        if (st.session_state.get("forecast_df") is not None and 
            st.session_state.get("forecast_sensor_name") == sensor_col and
            st.session_state.get("forecast_thresholds") is not None):
        
            df_forecast = st.session_state["forecast_df"]
            upper_threshold, lower_threshold = st.session_state["forecast_thresholds"]

            train_idx = st.session_state.train.index
            test_idx = st.session_state.test.index
            train_sensor = det.loc[det.index.intersection(train_idx)]
            test_sensor = det.loc[det.index.intersection(test_idx)]
            hist_anom = det[det["anomaly"] == 1]
            forecast_anom = df_forecast[df_forecast["anomaly"] == 1]
            train_end = train_idx[-1]
            last_ts = det.index[-1]

            fig = go.Figure()
            hover_template = (
                "Time: %{x|%Y-%m-%d %H:%M:%S}<br>"
                f"{sensor_col}: %{{y:.3f}}<br>"
                "Status: %{customdata}<extra>%{fullData.name}</extra>"
            )
            for sensor_data, label, color, line_width in [
                (train_sensor, "Training Data", "steelblue", 1),
                (test_sensor, "Testing Data", "green", 1.5),
            ]:
                status = sensor_data["anomaly"].map({0: "Normal", 1: "Anomaly"}).to_numpy()
                fig.add_trace(go.Scatter(
                    x=sensor_data.index,
                    y=sensor_data[sensor_col],
                    customdata=status,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=line_width),
                    marker=dict(size=4),
                    hovertemplate=hover_template,
                ))

            fig.add_trace(go.Scatter(
                x=hist_anom.index,
                y=hist_anom[sensor_col],
                mode="markers",
                name="Historical Anomaly",
                marker=dict(color="red", size=8),
                hovertemplate=(
                    "Time: %{x|%Y-%m-%d %H:%M:%S}<br>"
                    f"{sensor_col}: %{{y:.3f}}<br>"
                    "Status: Anomaly<extra>Historical Anomaly</extra>"
                ),
            ))

            fig.add_trace(go.Scatter(
                x=df_forecast.index,
                y=df_forecast["upper_ci"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=df_forecast.index,
                y=df_forecast["lower_ci"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255, 165, 0, 0.20)",
                name="95% Confidence Interval",
                hoverinfo="skip",
            ))

            forecast_status = df_forecast["anomaly"].map(
                {0: "Predicted normal", 1: "Predicted anomaly"}
            ).to_numpy()
            fig.add_trace(go.Scatter(
                x=df_forecast.index,
                y=df_forecast["forecast"],
                customdata=forecast_status,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="orange", width=2.5, dash="dash"),
                marker=dict(size=5),
                hovertemplate=(
                    "Time: %{x|%Y-%m-%d %H:%M:%S}<br>"
                    f"Forecast {sensor_col}: %{{y:.3f}}<br>"
                    "Status: %{customdata}<extra>Forecast</extra>"
                ),
            ))

            if len(forecast_anom):
                fig.add_trace(go.Scatter(
                    x=forecast_anom.index,
                    y=forecast_anom["forecast"],
                    mode="markers",
                    name="Forecast Anomaly / Failure",
                    marker=dict(color="darkred", symbol="diamond", size=11, line=dict(color="red", width=2)),
                    hovertemplate=(
                        "Time: %{x|%Y-%m-%d %H:%M:%S}<br>"
                        f"Forecast {sensor_col}: %{{y:.3f}}<br>"
                        "Status: Predicted anomaly<extra>Forecast Anomaly / Failure</extra>"
                    ),
                ))
                first_failure = forecast_anom.iloc[0]
                fig.add_annotation(
                    x=first_failure.name,
                    y=first_failure["forecast"],
                    text="Failure",
                    showarrow=True,
                    arrowhead=2,
                    ax=30,
                    ay=-30,
                    font=dict(size=13, color="red"),
                    arrowcolor="red",
                )

            y_values = pd.concat([
                det[sensor_col],
                df_forecast["forecast"],
                df_forecast["upper_ci"],
                df_forecast["lower_ci"],
            ])
            ymin, ymax = y_values.min(), y_values.max()
            fig.add_hline(
                y=upper_threshold,
                line_color="crimson",
                line_dash="dash",
                line_width=2,
                annotation_text=f"Upper Threshold ({upper_threshold:.2f})",
            )
            fig.add_hline(
                y=lower_threshold,
                line_color="darkgreen",
                line_dash="dash",
                line_width=2,
                annotation_text=f"Lower Threshold ({lower_threshold:.2f})",
            )
            if upper_threshold < ymax:
                fig.add_hrect(y0=upper_threshold, y1=ymax, fillcolor="red", opacity=0.08, line_width=0, layer="below")
            if lower_threshold > ymin:
                fig.add_hrect(y0=ymin, y1=lower_threshold, fillcolor="red", opacity=0.08, line_width=0, layer="below")
            fig.add_vline(x=train_end, line_color="purple", line_dash="dot", line_width=2, annotation_text="Train/Test Split")
            fig.add_vline(x=last_ts, line_color="black", line_dash="dot", line_width=2, annotation_text="Forecast Start")
            fig.update_layout(
                title=f"{sensor_col} Sensor Value — Training, Testing and Forecast with Failure Prediction",
                xaxis_title="Time",
                yaxis_title=f"{sensor_col} Sensor Value",
                hovermode="closest",
                height=560,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(t=100),
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)")
            st.plotly_chart(fig, width="stretch")

        else:
            st.info("Set parameters and click **Generate Forecast**.")

    # --------------------------------------------------------------------------
    # MAIN PAGE — ANOMALY EXPLANATION TABLE
    # --------------------------------------------------------------------------
    st.header("📋 Anomaly Explanation")

    if st.session_state.det_result is not None and "anomaly" in st.session_state.det_result.columns:
        det = st.session_state.det_result

        exp_sensor = st.selectbox("Sensor for explanation", feature_columns, key="exp_sensor_main")
    
        # Use fixed default parameters
        anomaly_table = build_anomaly_table(det, exp_sensor, window_size=5, spike_thr=2.5, slope_thr=0.3)
        st.metric("Anomalies explained", len(anomaly_table))
        st.dataframe(anomaly_table, use_container_width=True, height=500)
        if len(anomaly_table):
            st.download_button(
                "⬇️ Download explanation table (CSV)",
                data=anomaly_table.to_csv(index=False).encode("utf-8"),
                file_name=f"{exp_sensor}_anomaly_explanations.csv",
                mime="text/csv",
            )
    else:
        st.info("Run anomaly detection and apply a threshold first.")
with performance_tab:
    # ----------------------------------------------------------------------
    # MODEL PERFORMANCE COMPARISON
    # ----------------------------------------------------------------------
    st.header("📊 Model Performance Comparison")
    st.caption(
        "Compare the three anomaly-detection models and the three failure-prediction "
        "models using the same chronological train/test split."
    )

    st.subheader("1. Anomaly Detection Models")
    st.markdown(
        """
        **Evaluation guide:** A higher ASOI indicates greater separation between
        the score distributions treated as normal and anomalous by the ASOI
        calculation. It is a comparison indicator, not accuracy, precision,
        recall, or proof of actual failures because this dataset has no
        ground-truth failure labels. The anomaly rate shown here uses a common
        5th-percentile score cutoff for comparison.
        """
    )

    anomaly_perf = st.session_state.get("anomaly_comparison")
    if anomaly_perf is not None and not anomaly_perf.empty:
        anomaly_display = anomaly_perf.copy()
        anomaly_display.index.name = "Model"
        st.dataframe(
            anomaly_display[[
                c for c in [
                    "ASOI", "Anomaly Rate (%)", "Fit Time (seconds)",
                    "Raw Gap", "Mu Normal", "Mu Anomaly"
                ] if c in anomaly_display.columns
            ]].style.format(precision=4),
            use_container_width=True
        )
        if "ASOI" in anomaly_display.columns:
            chart_df = anomaly_display[["ASOI"]].sort_values("ASOI", ascending=False)
            st.bar_chart(chart_df, y="ASOI", use_container_width=True)
        best_anomaly = anomaly_display["ASOI"].idxmax()
        st.success(f"Best anomaly model by ASOI: **{best_anomaly}**")
    else:
        st.info("Click **Auto Find Best** under Anomaly Detection in the sidebar to evaluate all three models.")

    st.divider()
    st.subheader("2. Failure Prediction Models")
    st.markdown(
        """
        **Evaluation guide:**
        - **MAE:** average absolute prediction error; lower is better.
        - **RMSE:** penalises larger errors more strongly; lower is better.
        - **R²:** proportion of target-score variation explained by the model;
          closer to 1 is generally better.
        - **Fit Time:** time taken to train the model; lower is faster.

        The prediction target in this dashboard is the anomaly score, not a
        confirmed failure label. Therefore, these metrics evaluate prediction
        of anomaly scores rather than the probability of a real component failure.
        """
    )

    prediction_perf = st.session_state.get("prediction_comparison")
    if prediction_perf is not None and not prediction_perf.empty:
        prediction_display = prediction_perf.copy()
        prediction_display.index.name = "Model"
        st.dataframe(
            prediction_display[[
                c for c in ["MAE", "RMSE", "R2", "Fit Time (seconds)"]
                if c in prediction_display.columns
            ]].style.format(precision=4),
            use_container_width=True
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            best_mae = prediction_display["MAE"].idxmin()
            st.metric("Best by MAE", best_mae, f"{prediction_display.loc[best_mae, 'MAE']:.4f}")
        with metric_col2:
            best_rmse = prediction_display["RMSE"].idxmin()
            st.metric("Best by RMSE", best_rmse, f"{prediction_display.loc[best_rmse, 'RMSE']:.4f}")
        with metric_col3:
            best_r2 = prediction_display["R2"].idxmax()
            st.metric("Best by R²", best_r2, f"{prediction_display.loc[best_r2, 'R2']:.4f}")

        plot_metrics = prediction_display[["MAE", "RMSE"]].sort_values("RMSE")
        st.markdown("**Error comparison (lower is better)**")
        st.bar_chart(plot_metrics, use_container_width=True)
        st.markdown("**R² comparison (higher is better)**")
        st.bar_chart(prediction_display[["R2"]], use_container_width=True)
        st.success(f"Best failure-prediction model by RMSE: **{best_rmse}**")
    else:
        st.info("Click **Find Best Prediction** under Failure Prediction in the sidebar to evaluate all three models.")

    st.divider()
    st.caption(
        "For a fair comparison, keep the same selected sensor features and "
        "Train/Test Split while running all models. Changing either setting "
        "means the models are being evaluated on a different experiment."
    )

st.sidebar.markdown("---")
st.sidebar.caption("Anomaly Detection & Failure Prediction Dashboard · built with Streamlit")
