"""
Railway Critical Component — Anomaly Detection & Failure Prediction Dashboard
==============================================================================
Built to accompany: "Anomaly Detection and Failure Prediction Using Machine
Learning in Railway for Critical Component"

Run with:
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# ANOMALY MODELS (classic + autoencoder)
# ----------------------------------------------------------------------
ANOMALY_MODELS = [
    "Isolation Forest",
    "Local Outlier Factor",
    "One-Class SVM",
    "Elliptic Envelope (Robust Covariance)",
    "Gaussian Mixture Model",
    "PCA Reconstruction Error",
    "Autoencoder (Deep Learning)",          # <-- ADDED
]

# ----------------------------------------------------------------------
# FAILURE PREDICTION MODELS (classic + time series + deep learning)
# ----------------------------------------------------------------------
REGRESSION_MODELS = [
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "Support Vector Regression",
    "Ridge Regression",
    "K-Nearest Neighbors",
    "ARIMA (Time Series)",                  # <-- ADDED
    "LSTM (Deep Learning)",                 # <-- ADDED
]
if XGB_AVAILABLE:
    REGRESSION_MODELS.insert(2, "XGBoost")

st.set_page_config(
    page_title="Railway Anomaly Detection & Failure Prediction",
    page_icon="🚆",
    layout="wide",
)

# --------------------------------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------------------------------
for key in ["df_raw", "df", "feature_columns", "train", "test", "scaler",
            "det_result", "det_score_col", "det_threshold", "det_model_name",
            "pred_result", "native_threshold", "y_scaler"]:
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
    """Return (binary anomaly flag, threshold value) given a scoring method."""
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


def get_anomaly_scores(model_name, X_train, X_test, params):
    """
    Fit chosen anomaly-detection model and return train+test scores (higher = more normal).
    Also returns the native threshold (if any) to be used with 'Auto' thresholding.
    """
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
        model = OneClassSVM(
            kernel="rbf",
            nu=params["nu"],
            gamma="scale",
        )
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
            return -err  # higher = normal

        train_scores = _neg_recon_error(X_train)
        test_scores = _neg_recon_error(X_test)
        native_threshold = np.percentile(train_scores, params["contamination"] * 100)

    # ----- AUTOENCODER (Deep Learning) -----
    elif model_name == "Autoencoder (Deep Learning)":
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Please run: pip install tensorflow")
        seq_len = params["seq_len"]
        n_features = X_train.shape[1]

        def create_sequences(data, seq_len):
            X_seq = []
            for i in range(len(data) - seq_len + 1):
                X_seq.append(data[i:i+seq_len])
            return np.array(X_seq)

        X_train_seq = create_sequences(X_train.values, seq_len)
        model = Sequential()
        model.add(LSTM(params["units"], activation='relu', input_shape=(seq_len, n_features)))
        model.add(RepeatVector(seq_len))
        model.add(LSTM(params["units"], activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(n_features)))
        model.compile(optimizer='adam', loss='mse')
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0)
        model.fit(X_train_seq, X_train_seq,
                  epochs=params["epochs"],
                  batch_size=params["batch_size"],
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


def get_regressor(model_name, params):
    """Return a regressor object (classic, ARIMA, or LSTM)."""
    # Classic ML models
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "Support Vector Regression":
        return SVR(kernel="rbf", C=params["C"], epsilon=params["epsilon"], gamma="scale")
    if model_name == "Extra Trees":
        return ExtraTreesRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=42,
        )
    if model_name == "Ridge Regression":
        return Ridge(alpha=params["alpha"], random_state=42)
    if model_name == "K-Nearest Neighbors":
        return KNeighborsRegressor(n_neighbors=params["n_neighbors"], weights=params["weights"])

    # ARIMA – we return a tuple (model, training_data) to be fitted later
    if model_name == "ARIMA (Time Series)":
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is not installed. Please run: pip install statsmodels")
        # We'll store the order and return a placeholder; the actual fit happens in the training step
        return ("ARIMA", params)

    # LSTM – placeholder, training happens in the main loop
    if model_name == "LSTM (Deep Learning)":
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Please run: pip install tensorflow")
        return ("LSTM", params)

    raise ValueError(f"Unknown regressor: {model_name}")


def classify_anomaly(sensor_col, val, prev_val, window_vals, spike_thr, slope_thr):
    """Rule-based explanation for a flagged anomaly point."""
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
    """Build a human-readable explanation table for every flagged anomaly."""
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
    """Simple linear-trend extrapolation forecast with a 95% confidence band."""
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
# SIDEBAR — DATA UPLOAD
# --------------------------------------------------------------------------
st.sidebar.title("🚆 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload sensor dataset (Excel .xlsx)", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df_raw = load_excel(uploaded_file)
        st.session_state.df_raw = df_raw
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")

st.title("🚆 Railway Critical Component — Anomaly Detection & Failure Prediction")
st.caption("7 anomaly‑detection models · 8 failure‑prediction models · configurable thresholding · forecast & failure‑zone chart · anomaly explanation table")

if st.session_state.df_raw is None:
    st.info("👈 Upload an Excel file with sensor readings to get started. A timestamp column and one or more numeric sensor columns are expected.")
    st.stop()

df_raw = st.session_state.df_raw

# --------------------------------------------------------------------------
# STEP 1 — CONFIGURE COLUMNS
# --------------------------------------------------------------------------
st.header("1️⃣ Data Setup")

col_a, col_b = st.columns([1, 2])
with col_a:
    cols = df_raw.columns.tolist()
    ts_guess = next((c for c in cols if "time" in c.lower() or "date" in c.lower()), cols[0])
    ts_col = st.selectbox("Timestamp column", cols, index=cols.index(ts_guess))

df = df_raw.copy()
try:
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()
except Exception as e:
    st.error(f"Could not parse '{ts_col}' as a timestamp: {e}")
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
with col_b:
    feature_columns = st.multiselect(
        "Sensor / feature columns to use", numeric_cols, default=numeric_cols
    )

if not feature_columns:
    st.warning("Select at least one numeric feature column.")
    st.stop()

st.session_state.feature_columns = feature_columns
numeric_df = df[feature_columns].dropna()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{numeric_df.shape[0]:,}")
c2.metric("Features", len(feature_columns))
c3.metric("Missing (dropped)", int(df[feature_columns].isnull().any(axis=1).sum()))
c4.metric("Duplicate rows", int(df.duplicated().sum()))

with st.expander("Preview data"):
    st.dataframe(df_raw.head(20), use_container_width=True)

with st.expander("Sensor trends & correlation"):
    fig = px.line(numeric_df, x=numeric_df.index, y=feature_columns, title="Sensor Trends Over Time")
    st.plotly_chart(fig, use_container_width=True)
    corr = numeric_df.corr()
    fig2 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Matrix")
    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------------------
# STEP 2 — TRAIN/TEST SPLIT + SCALING
# --------------------------------------------------------------------------
st.header("2️⃣ Train / Test Split & Scaling")
split_pct = st.slider("Training set size (%)", 50, 95, 80)
train_size = int(len(numeric_df) * split_pct / 100)
train_raw = numeric_df.iloc[:train_size].copy()
test_raw = numeric_df.iloc[train_size:].copy()

scaler = StandardScaler().fit(train_raw[feature_columns])
train = pd.DataFrame(scaler.transform(train_raw[feature_columns]), columns=feature_columns, index=train_raw.index)
test = pd.DataFrame(scaler.transform(test_raw[feature_columns]), columns=feature_columns, index=test_raw.index)
st.session_state.train, st.session_state.test, st.session_state.scaler = train, test, scaler

st.caption(f"Train: {train.shape[0]:,} rows  |  Test: {test.shape[0]:,} rows")

# --------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔎 Anomaly Detection",
    "📈 Failure Prediction",
    "🔮 Forecast & Failure Zone",
    "📋 Anomaly Explanation",
    "⬇️ Export Results",
])

# ==========================================================================
# TAB 1 — ANOMALY DETECTION
# ==========================================================================
with tab1:
    st.subheader("Model & Hyperparameters")
    model_name = st.selectbox("Anomaly detection model", ANOMALY_MODELS)

    params = {}
    p1, p2, p3 = st.columns(3)

    # Classic models hyperparams
    if model_name == "Isolation Forest":
        with p1:
            params["n_estimators"] = st.slider("n_estimators", 50, 500, 200, 50)
        with p2:
            params["contamination"] = st.slider("contamination", 0.01, 0.20, 0.04, 0.01)
        with p3:
            params["max_features"] = st.slider("max_features", 0.1, 1.0, 0.7, 0.1)
    elif model_name == "Local Outlier Factor":
        with p1:
            params["n_neighbors"] = st.slider("n_neighbors", 5, 50, 20, 1)
        with p2:
            params["contamination"] = st.slider("contamination", 0.01, 0.20, 0.04, 0.01)
    elif model_name == "One-Class SVM":
        with p1:
            params["nu"] = st.slider("nu (expected anomaly fraction)", 0.01, 0.20, 0.04, 0.01)
    elif model_name == "Elliptic Envelope (Robust Covariance)":
        with p1:
            params["contamination"] = st.slider("contamination", 0.01, 0.20, 0.04, 0.01, key="ee_c")
        with p2:
            params["support_fraction"] = st.slider("support_fraction", 0.5, 1.0, 0.8, 0.05, key="ee_sf")
    elif model_name == "Gaussian Mixture Model":
        with p1:
            params["n_components"] = st.slider("n_components (normal-behaviour clusters)", 1, 10, 3, 1, key="gmm_k")
        with p2:
            params["contamination"] = st.slider("contamination", 0.01, 0.20, 0.04, 0.01, key="gmm_c")
    elif model_name == "PCA Reconstruction Error":
        with p1:
            params["n_components"] = st.slider("n_components (retained variance dims)", 1, max(2, len(feature_columns) - 1), min(3, max(1, len(feature_columns) - 1)), 1, key="pca_k")
        with p2:
            params["contamination"] = st.slider("contamination", 0.01, 0.20, 0.04, 0.01, key="pca_c")
    # ----- Autoencoder hyperparams -----
    elif model_name == "Autoencoder (Deep Learning)":
        with p1:
            params["seq_len"] = st.slider("Sequence length", 2, 20, 10, key="ae_seq")
        with p2:
            params["units"] = st.slider("LSTM units", 8, 128, 32, key="ae_units")
        with p3:
            params["epochs"] = st.slider("Epochs", 10, 100, 50, key="ae_epochs")
        with p1:
            params["batch_size"] = st.slider("Batch size", 8, 128, 32, key="ae_batch")
        with p2:
            params["contamination"] = st.slider("contamination", 0.01, 0.20, 0.04, 0.01, key="ae_c")
        if not TF_AVAILABLE:
            st.warning("⚠️ TensorFlow is not installed. Please run: `pip install tensorflow`")

    run_detect = st.button("🚀 Fit anomaly detection model", type="primary")

    if run_detect:
        with st.spinner("Fitting model..."):
            try:
                model, train_scores, test_scores, native_thr = get_anomaly_scores(
                    model_name, train[feature_columns], test[feature_columns], params
                )
            except ImportError as e:
                st.error(f"❌ Missing dependency: {e}")
                st.stop()
        full_scores = pd.Series(
            np.concatenate([train_scores, test_scores]),
            index=list(train.index) + list(test.index),
            name="anomaly_score",
        ).sort_index()

        result = numeric_df.loc[full_scores.index].copy()
        result["anomaly_score"] = full_scores
        result["split"] = ["train"] * len(train_scores) + ["test"] * len(test_scores)

        st.session_state.det_result = result
        st.session_state.det_model_name = model_name
        st.session_state.native_threshold = native_thr
        st.success(f"{model_name} fitted on {len(feature_columns)} features. Lower score = more anomalous.")

    if st.session_state.det_result is not None:
        result = st.session_state.det_result

        st.subheader("Threshold Method")
        # ----- THRESHOLD SELECTION WITH EXPLANATION (ℹ️) -----
        thr_method = st.selectbox(
            "Choose how the anomaly threshold is determined",
            ["Auto (model's native threshold)", "Percentile (worst X%)", "Mean - K*Std", "Manual value"],
            help="Score convention: lower anomaly_score = more anomalous. 'Auto' uses the model's built‑in threshold (if available)."
        )

        # ----- Compact explanation expander -----
        with st.expander("ℹ️ What does each threshold method mean?"):
            st.markdown("""
            - **Auto (model's native threshold)** – Uses the threshold the model learned internally (e.g., Isolation Forest's `offset`).  
              Use this when you trust the model's own sensitivity.

            - **Percentile (worst X%)** – Flags the lowest X% of scores. For example, 5% means the 5% most anomalous points are marked.  
              Simple and lets you control exactly how many points are flagged.

            - **Mean - K*Std** – Flags points whose score is **K standard deviations** below the average.  
              Works well if scores are roughly bell‑shaped (normal distribution). Good for catching extreme outliers.

            - **Manual value** – You set a fixed number. Any score below that is an anomaly.  
              Use this when you have domain knowledge and know the danger limit.
            """)

        thr_kwargs = {}
        if thr_method == "Percentile (worst X%)":
            thr_kwargs["percentile"] = st.slider("Flag the worst X% of scores", 1, 25, 5)
        elif thr_method == "Mean - K*Std":
            thr_kwargs["k"] = st.slider("K (number of std deviations below mean)", 0.5, 4.0, 2.0, 0.5)
        elif thr_method == "Manual value":
            smin, smax = float(result["anomaly_score"].min()), float(result["anomaly_score"].max())
            thr_kwargs["manual_value"] = st.slider("Manual threshold value", smin, smax, float(result["anomaly_score"].quantile(0.05)))

        native_thr = st.session_state.native_threshold if st.session_state.native_threshold is not None else None

        try:
            flag, thr_value = apply_threshold(
                result["anomaly_score"],
                thr_method,
                native_thr=native_thr,
                **thr_kwargs
            )
            result["anomaly"] = flag
            st.session_state.det_result = result
            st.session_state.det_threshold = thr_value

            m1, m2, m3 = st.columns(3)
            m1.metric("Threshold value", f"{thr_value:.4f}")
            m2.metric("Anomalies flagged", int(result["anomaly"].sum()))
            m3.metric("Anomaly rate", f"{result['anomaly'].mean()*100:.2f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result.index, y=result["anomaly_score"], mode="lines", name="Anomaly score", line=dict(color="steelblue")))
            anom = result[result["anomaly"] == 1]
            fig.add_trace(go.Scatter(x=anom.index, y=anom["anomaly_score"], mode="markers", name="Flagged anomaly", marker=dict(color="red", size=6)))
            fig.add_hline(y=thr_value, line_dash="dash", line_color="red", annotation_text="Threshold")
            fig.update_layout(title=f"{st.session_state.det_model_name} — Anomaly Score Over Time", xaxis_title="Time", yaxis_title="Anomaly Score", height=450)
            st.plotly_chart(fig, use_container_width=True)

            fig_hist = px.histogram(result, x="anomaly_score", nbins=60, title="Anomaly Score Distribution")
            fig_hist.add_vline(x=thr_value, line_dash="dash", line_color="red")
            st.plotly_chart(fig_hist, use_container_width=True)

            with st.expander("Flagged anomaly rows"):
                st.dataframe(anom.sort_values("anomaly_score").head(200), use_container_width=True)
        except Exception as e:
            st.error(f"Error applying threshold: {e}")

    else:
        st.info("Configure hyperparameters and click **Fit anomaly detection model** to continue.")

# ==========================================================================
# TAB 2 — FAILURE PREDICTION
# ==========================================================================
with tab2:
    if st.session_state.det_result is None or "anomaly" not in st.session_state.det_result.columns:
        st.info("Run **Anomaly Detection** first (Tab 1) — its anomaly score is used as the regression target for failure prediction.")
    else:
        st.subheader("Model & Hyperparameters")
        reg_name = st.selectbox("Failure prediction model", REGRESSION_MODELS)

        rparams = {}
        r1, r2, r3 = st.columns(3)

        # ----- Classic ML models -----
        if reg_name == "Random Forest":
            with r1:
                rparams["n_estimators"] = st.slider("n_estimators", 50, 600, 200, 50, key="rf_n")
            with r2:
                depth = st.slider("max_depth (0 = None)", 0, 50, 20, key="rf_d")
                rparams["max_depth"] = None if depth == 0 else depth
        elif reg_name == "Extra Trees":
            with r1:
                rparams["n_estimators"] = st.slider("n_estimators", 50, 600, 200, 50, key="et_n")
            with r2:
                depth = st.slider("max_depth (0 = None)", 0, 50, 20, key="et_d")
                rparams["max_depth"] = None if depth == 0 else depth
        elif reg_name == "XGBoost":
            with r1:
                rparams["n_estimators"] = st.slider("n_estimators", 50, 600, 200, 50, key="xgb_n")
            with r2:
                rparams["max_depth"] = st.slider("max_depth", 2, 15, 6, key="xgb_d")
            with r3:
                rparams["learning_rate"] = st.slider("learning_rate", 0.01, 0.5, 0.1, 0.01, key="xgb_lr")
        elif reg_name == "Gradient Boosting":
            with r1:
                rparams["n_estimators"] = st.slider("n_estimators", 50, 600, 200, 50, key="gb_n")
            with r2:
                rparams["max_depth"] = st.slider("max_depth", 2, 15, 3, key="gb_d")
            with r3:
                rparams["learning_rate"] = st.slider("learning_rate", 0.01, 0.5, 0.1, 0.01, key="gb_lr")
        elif reg_name == "Support Vector Regression":
            with r1:
                rparams["C"] = st.slider("C", 0.1, 10.0, 2.0, 0.1, key="svr_c")
            with r2:
                rparams["epsilon"] = st.slider("epsilon", 0.001, 0.5, 0.01, 0.001, key="svr_eps")
        elif reg_name == "Ridge Regression":
            with r1:
                rparams["alpha"] = st.slider("alpha (regularisation strength)", 0.01, 10.0, 1.0, 0.01, key="ridge_a")
        elif reg_name == "K-Nearest Neighbors":
            with r1:
                rparams["n_neighbors"] = st.slider("n_neighbors", 2, 50, 10, 1, key="knn_n")
            with r2:
                rparams["weights"] = st.selectbox("weights", ["uniform", "distance"], key="knn_w")
        # ----- ARIMA (time series) -----
        elif reg_name == "ARIMA (Time Series)":
            with r1:
                rparams["p"] = st.slider("p (AR order)", 0, 10, 1, key="ar_p")
            with r2:
                rparams["d"] = st.slider("d (differencing)", 0, 2, 1, key="ar_d")
            with r3:
                rparams["q"] = st.slider("q (MA order)", 0, 10, 1, key="ar_q")
            if not STATSMODELS_AVAILABLE:
                st.warning("⚠️ statsmodels is not installed. Please run: `pip install statsmodels`")
        # ----- LSTM (deep learning) -----
        elif reg_name == "LSTM (Deep Learning)":
            with r1:
                rparams["lookback"] = st.slider("Lookback (time steps)", 2, 30, 10, key="lstm_look")
            with r2:
                rparams["units"] = st.slider("LSTM units", 8, 128, 32, key="lstm_units")
            with r3:
                rparams["epochs"] = st.slider("Epochs", 10, 100, 50, key="lstm_epochs")
            with r1:
                rparams["batch_size"] = st.slider("Batch size", 8, 128, 32, key="lstm_batch")
            if not TF_AVAILABLE:
                st.warning("⚠️ TensorFlow is not installed. Please run: `pip install tensorflow`")

        run_pred = st.button("🚀 Train failure prediction model", type="primary")

        if run_pred:
            det = st.session_state.det_result
            y_train = det.loc[train.index, "anomaly_score"]
            y_test = det.loc[test.index, "anomaly_score"]

            with st.spinner("Training prediction model..."):
                try:
                    # Get the regressor placeholder
                    reg_placeholder = get_regressor(reg_name, rparams)
                except ImportError as e:
                    st.error(f"❌ Missing dependency: {e}")
                    st.stop()

                # ----- Classic sklearn models -----
                if reg_name in ["Random Forest", "Extra Trees", "XGBoost", "Gradient Boosting",
                                "Support Vector Regression", "Ridge Regression", "K-Nearest Neighbors"]:
                    reg = reg_placeholder
                    reg.fit(train[feature_columns], y_train)
                    pred_train = reg.predict(train[feature_columns])
                    pred_test = reg.predict(test[feature_columns])
                    st.session_state["reg_model_obj"] = reg

                # ----- ARIMA -----
                elif reg_name == "ARIMA (Time Series)":
                    # Fit ARIMA on the training target
                    model = ARIMA(y_train, order=(rparams["p"], rparams["d"], rparams["q"]))
                    fitted = model.fit()
                    # In‑sample predictions
                    pred_train = fitted.predict(start=0, end=len(y_train)-1, dynamic=False)
                    # Forecast on test
                    pred_test = fitted.forecast(steps=len(y_test))
                    st.session_state["reg_model_obj"] = fitted

                # ----- LSTM (Deep Learning) -----
                elif reg_name == "LSTM (Deep Learning)":
                    # Scale the target
                    y_scaler = StandardScaler()
                    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
                    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
                    st.session_state["y_scaler"] = y_scaler

                    lookback = rparams["lookback"]
                    n_features = len(feature_columns)

                    def create_seq(X, y, lookback):
                        Xs, ys = [], []
                        for i in range(lookback, len(X)):
                            Xs.append(X[i-lookback:i])
                            ys.append(y[i])
                        return np.array(Xs), np.array(ys)

                    X_train_seq, y_train_seq = create_seq(train[feature_columns].values, y_train_scaled, lookback)
                    X_test_seq, y_test_seq = create_seq(test[feature_columns].values, y_test_scaled, lookback)

                    # Build LSTM
                    model = Sequential()
                    model.add(LSTM(rparams["units"], activation='relu', input_shape=(lookback, n_features)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0)
                    model.fit(X_train_seq, y_train_seq,
                              epochs=rparams["epochs"],
                              batch_size=rparams["batch_size"],
                              callbacks=[early_stop],
                              verbose=0)

                    # Predictions (scaled)
                    pred_train_scaled = model.predict(X_train_seq, verbose=0).flatten()
                    pred_test_scaled = model.predict(X_test_seq, verbose=0).flatten()

                    # Inverse transform
                    pred_train = y_scaler.inverse_transform(pred_train_scaled.reshape(-1, 1)).flatten()
                    pred_test = y_scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).flatten()

                    # Align with original indices (first `lookback` points have no prediction)
                    full_train_pred = np.full(len(y_train), np.nan)
                    full_train_pred[lookback:] = pred_train
                    pred_train = full_train_pred

                    full_test_pred = np.full(len(y_test), np.nan)
                    full_test_pred[lookback:] = pred_test
                    pred_test = full_test_pred

                    st.session_state["reg_model_obj"] = model

                # Combine results (remove NaNs for metrics)
                y_true_all = pd.concat([y_train, y_test])
                y_pred_all = np.concatenate([pred_train, pred_test])
                valid_mask = ~np.isnan(y_pred_all)
                y_true_valid = y_true_all[valid_mask]
                y_pred_valid = y_pred_all[valid_mask]

                mae = mean_absolute_error(y_true_valid, y_pred_valid)
                rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
                r2s = r2_score(y_true_valid, y_pred_valid)

                pred_df = pd.DataFrame({
                    "actual_score": y_true_all,
                    "predicted_score": y_pred_all,
                    "split": ["train"] * len(y_train) + ["test"] * len(y_test),
                }, index=y_true_all.index).sort_index()

                st.session_state.pred_result = pred_df
                st.session_state["reg_metrics"] = {"MAE": mae, "RMSE": rmse, "R2": r2s}
                st.session_state["reg_model_name"] = reg_name
                st.success(f"{reg_name} trained to predict anomaly score.")

        if st.session_state.pred_result is not None:
            pred_df = st.session_state.pred_result
            metrics = st.session_state["reg_metrics"]

            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{metrics['MAE']:.4f}")
            m2.metric("RMSE", f"{metrics['RMSE']:.4f}")
            m3.metric("R² Score", f"{metrics['R2']:.4f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["actual_score"], name="Actual score", line=dict(color="steelblue")))
            fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["predicted_score"], name="Predicted score", line=dict(color="orange", dash="dot")))
            if st.session_state.det_threshold is not None:
                fig.add_hline(y=st.session_state.det_threshold, line_dash="dash", line_color="red", annotation_text="Anomaly threshold")
            fig.update_layout(title=f"{st.session_state['reg_model_name']} — Actual vs Predicted Anomaly Score", xaxis_title="Time", yaxis_title="Score", height=450)
            st.plotly_chart(fig, use_container_width=True)

            fig_scatter = px.scatter(pred_df.dropna(), x="actual_score", y="predicted_score", color="split", title="Predicted vs Actual", opacity=0.6)
            minv = min(pred_df["actual_score"].min(), pred_df["predicted_score"].min())
            maxv = max(pred_df["actual_score"].max(), pred_df["predicted_score"].max())
            fig_scatter.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode="lines", line=dict(color="gray", dash="dash"), name="Ideal"))
            st.plotly_chart(fig_scatter, use_container_width=True)

            reg = st.session_state["reg_model_obj"]
            if hasattr(reg, "feature_importances_"):
                imp = pd.Series(reg.feature_importances_, index=feature_columns).sort_values(ascending=False)
                fig_imp = px.bar(imp, orientation="h", title="Feature Importance", labels={"value": "Importance", "index": "Feature"})
                st.plotly_chart(fig_imp, use_container_width=True)

            if st.session_state.det_threshold is not None:
                thr = st.session_state.det_threshold
                pred_df["predicted_failure"] = (pred_df["predicted_score"] <= thr).astype(int)
                rate = pred_df["predicted_failure"].mean() * 100
                st.metric("Predicted failure/anomaly rate (using Tab 1 threshold)", f"{rate:.2f}%")
                with st.expander("Rows predicted as failure/anomaly"):
                    st.dataframe(pred_df[pred_df["predicted_failure"] == 1].sort_values("predicted_score").head(200), use_container_width=True)

# ==========================================================================
# TAB 3 — FORECAST & FAILURE ZONE (TP2-style chart, generalised to any sensor)
# ==========================================================================
with tab3:
    if st.session_state.det_result is None or "anomaly" not in st.session_state.det_result.columns:
        st.info("Run **Anomaly Detection** first (Tab 1) so historical anomalies and thresholds are available.")
    else:
        det = st.session_state.det_result

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sensor_default = "TP2" if "TP2" in feature_columns else feature_columns[0]
            sensor_col = st.selectbox("Sensor to plot", feature_columns, index=feature_columns.index(sensor_default))
        with fc2:
            horizon = st.slider("Forecast horizon (steps ahead)", 5, 500, 168)
        with fc3:
            trend_window = st.slider("Trend window (points used to fit the trend)", 10, 500, 100)

        run_forecast = st.button("🚀 Generate forecast & failure zone chart", type="primary")

        if run_forecast:
            normal_vals = det[det["anomaly"] == 0][sensor_col]
            upper_threshold = normal_vals.max()
            lower_threshold = normal_vals.min()

            df_forecast = build_linear_forecast(det[sensor_col], horizon, trend_window)
            df_forecast["anomaly"] = (
                (df_forecast["forecast"] > upper_threshold) | (df_forecast["forecast"] < lower_threshold)
            ).astype(int)

            st.session_state["forecast_df"] = df_forecast
            st.session_state["forecast_sensor"] = sensor_col
            st.session_state["forecast_thresholds"] = (upper_threshold, lower_threshold)

        if st.session_state.get("forecast_df") is not None and st.session_state.get("forecast_sensor") == sensor_col:
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

            fig, ax = plt.subplots(figsize=(16, 7))

            ax.plot(train_sensor.index, train_sensor[sensor_col], color="steelblue", linewidth=1, label="Training Data")
            ax.plot(test_sensor.index, test_sensor[sensor_col], color="green", linewidth=1.5, label="Testing Data")
            ax.scatter(hist_anom.index, hist_anom[sensor_col], color="red", s=35, zorder=6, label="Historical Anomaly")

            ax.plot(df_forecast.index, df_forecast["forecast"], color="orange", linewidth=2.5, linestyle="--", label="Forecast")
            ax.fill_between(df_forecast.index, df_forecast["lower_ci"], df_forecast["upper_ci"], color="orange", alpha=0.20, label="95% Confidence Interval")

            if len(forecast_anom):
                ax.scatter(forecast_anom.index, forecast_anom["forecast"], color="darkred", marker="D", s=90, zorder=8, label="Forecast Anomaly")
                ax.scatter(forecast_anom.index, forecast_anom["forecast"], s=400, facecolors="none", edgecolors="red", linewidths=2, zorder=9, label="Failure")
                first_failure = forecast_anom.iloc[0]
                ax.annotate(
                    "Failure",
                    xy=(first_failure.name, first_failure["forecast"]),
                    xytext=(30, 30),
                    textcoords="offset points",
                    fontsize=13,
                    fontweight="bold",
                    color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=2),
                )

            ax.axhline(upper_threshold, color="crimson", linestyle="--", linewidth=2, label=f"Upper Threshold ({upper_threshold:.2f})")
            ax.axhline(lower_threshold, color="darkgreen", linestyle="--", linewidth=2, label=f"Lower Threshold ({lower_threshold:.2f})")

            ymin, ymax = ax.get_ylim()
            ax.axhspan(upper_threshold, ymax, color="red", alpha=0.08)
            ax.axhspan(ymin, lower_threshold, color="red", alpha=0.08)

            ax.axvline(train_end, color="purple", linestyle=":", linewidth=2, label="Train/Test Split")
            ax.axvline(last_ts, color="black", linestyle=":", linewidth=2, label="Forecast Start")

            ax.set_title(f"{sensor_col} Sensor Value — Training, Testing and Forecast with Failure Prediction", fontsize=16, fontweight="bold", pad=15)
            ax.set_xlabel("Time", fontsize=12, fontweight="bold")
            ax.set_ylabel(f"{sensor_col} Sensor Value", fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=9, ncol=2, frameon=True, edgecolor="black")
            fig.tight_layout()

            st.pyplot(fig)

            m1, m2, m3 = st.columns(3)
            m1.metric("Upper threshold", f"{upper_threshold:.3f}")
            m2.metric("Lower threshold", f"{lower_threshold:.3f}")
            m3.metric("Forecast points flagged as failure", int(df_forecast["anomaly"].sum()))
        else:
            st.info("Set the parameters and click **Generate forecast & failure zone chart**.")

# ==========================================================================
# TAB 4 — ANOMALY EXPLANATION TABLE
# ==========================================================================
with tab4:
    if st.session_state.det_result is None or "anomaly" not in st.session_state.det_result.columns:
        st.info("Run **Anomaly Detection** first (Tab 1) to generate the explanation table.")
    else:
        det = st.session_state.det_result

        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            exp_sensor = st.selectbox("Sensor column", feature_columns, key="exp_sensor")
        with ec2:
            exp_window = st.slider("Window size", 2, 20, 5, key="exp_window")
        with ec3:
            default_spike = float(round(det[exp_sensor].std() * 1.5, 2)) or 2.5
            exp_spike = st.number_input("Spike threshold (Δ)", value=default_spike, step=0.1, key="exp_spike")
        with ec4:
            default_slope = float(round(det[exp_sensor].std() * 0.3, 2)) or 0.3
            exp_slope = st.number_input("Slope threshold", value=default_slope, step=0.05, key="exp_slope")

        anomaly_table = build_anomaly_table(det, exp_sensor, exp_window, exp_spike, exp_slope)

        st.metric("Anomalies explained", len(anomaly_table))
        st.dataframe(anomaly_table, use_container_width=True, height=500)

        if len(anomaly_table):
            st.download_button(
                "⬇️ Download explanation table (CSV)",
                data=anomaly_table.to_csv(index=False).encode("utf-8"),
                file_name=f"{exp_sensor}_anomaly_explanations.csv",
                mime="text/csv",
            )

# ==========================================================================
# TAB 5 — EXPORT
# ==========================================================================
with tab5:
    st.subheader("Download combined results")
    if st.session_state.det_result is None:
        st.info("Run anomaly detection (and optionally failure prediction) first.")
    else:
        export_df = st.session_state.det_result.copy()
        if st.session_state.pred_result is not None:
            export_df = export_df.join(
                st.session_state.pred_result[["predicted_score"]], how="left"
            )
            if "predicted_failure" in st.session_state.pred_result.columns:
                export_df = export_df.join(
                    st.session_state.pred_result[["predicted_failure"]], how="left"
                )

        st.dataframe(export_df.head(50), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Download as CSV",
                data=export_df.to_csv(index=True).encode("utf-8"),
                file_name="anomaly_failure_results.csv",
                mime="text/csv",
            )
        with c2:
            st.download_button(
                "⬇️ Download as Excel",
                data=to_excel_bytes(export_df),
                file_name="anomaly_failure_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

st.sidebar.markdown("---")
st.sidebar.caption("Anomaly Detection & Failure Prediction Dashboard · built with Streamlit")