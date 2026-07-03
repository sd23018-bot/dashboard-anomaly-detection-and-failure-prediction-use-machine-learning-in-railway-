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
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM, SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

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
            "pred_result"]:
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


def apply_threshold(scores: pd.Series, method: str, **kwargs) -> tuple[pd.Series, float]:
    """Return (binary anomaly flag, threshold value) given a scoring method."""
    if method == "Percentile (worst X%)":
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
    """Fit chosen anomaly-detection model and return train+test scores (higher = more normal)."""
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

    else:
        raise ValueError("Unknown model")

    return model, train_scores, test_scores, native_threshold


def get_regressor(model_name, params):
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
    raise ValueError("Unknown regressor")


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
st.caption("Isolation Forest / LOF / One-Class SVM for anomaly scoring · Random Forest / XGBoost / SVR for failure (score) prediction · configurable thresholding")

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
tab1, tab2, tab3 = st.tabs(["🔎 Anomaly Detection", "📈 Failure Prediction (Regression)", "⬇️ Export Results"])

# ==========================================================================
# TAB 1 — ANOMALY DETECTION
# ==========================================================================
with tab1:
    st.subheader("Model & Hyperparameters")
    model_name = st.selectbox(
        "Anomaly detection model", ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"]
    )

    params = {}
    p1, p2, p3 = st.columns(3)
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
    else:  # One-Class SVM
        with p1:
            params["nu"] = st.slider("nu (expected anomaly fraction)", 0.01, 0.20, 0.04, 0.01)

    run_detect = st.button("🚀 Fit anomaly detection model", type="primary")

    if run_detect:
        with st.spinner("Fitting model..."):
            model, train_scores, test_scores, native_thr = get_anomaly_scores(
                model_name, train[feature_columns], test[feature_columns], params
            )
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
        st.session_state["native_threshold"] = native_thr
        st.success(f"{model_name} fitted on {len(feature_columns)} features. Lower score = more anomalous.")

    if st.session_state.det_result is not None:
        result = st.session_state.det_result

        st.subheader("Threshold Method")
        thr_method = st.selectbox(
            "Choose how the anomaly threshold is determined",
            ["Percentile (worst X%)", "Mean - K*Std", "Manual value"],
            help="Score convention: lower anomaly_score = more anomalous. Anything at/below the threshold is flagged.",
        )
        thr_kwargs = {}
        if thr_method == "Percentile (worst X%)":
            thr_kwargs["percentile"] = st.slider("Flag the worst X% of scores", 1, 25, 5)
        elif thr_method == "Mean - K*Std":
            thr_kwargs["k"] = st.slider("K (number of std deviations below mean)", 0.5, 4.0, 2.0, 0.5)
        else:
            smin, smax = float(result["anomaly_score"].min()), float(result["anomaly_score"].max())
            thr_kwargs["manual_value"] = st.slider("Manual threshold value", smin, smax, float(result["anomaly_score"].quantile(0.05)))

        flag, thr_value = apply_threshold(result["anomaly_score"], thr_method, **thr_kwargs)
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
    else:
        st.info("Configure hyperparameters and click **Fit anomaly detection model** to continue.")

# ==========================================================================
# TAB 2 — FAILURE PREDICTION (REGRESSION)
# ==========================================================================
with tab2:
    if st.session_state.det_result is None or "anomaly" not in st.session_state.det_result.columns:
        st.info("Run **Anomaly Detection** first (Tab 1) — its anomaly score is used as the regression target for failure prediction.")
    else:
        st.subheader("Model & Hyperparameters")
        reg_options = ["Random Forest", "Support Vector Regression"]
        if XGB_AVAILABLE:
            reg_options.insert(1, "XGBoost")
        reg_name = st.selectbox("Failure prediction (regression) model", reg_options)

        rparams = {}
        r1, r2, r3 = st.columns(3)
        if reg_name == "Random Forest":
            with r1:
                rparams["n_estimators"] = st.slider("n_estimators", 50, 600, 200, 50, key="rf_n")
            with r2:
                depth = st.slider("max_depth (0 = None)", 0, 50, 20, key="rf_d")
                rparams["max_depth"] = None if depth == 0 else depth
        elif reg_name == "XGBoost":
            with r1:
                rparams["n_estimators"] = st.slider("n_estimators", 50, 600, 200, 50, key="xgb_n")
            with r2:
                rparams["max_depth"] = st.slider("max_depth", 2, 15, 6, key="xgb_d")
            with r3:
                rparams["learning_rate"] = st.slider("learning_rate", 0.01, 0.5, 0.1, 0.01, key="xgb_lr")
        else:
            with r1:
                rparams["C"] = st.slider("C", 0.1, 10.0, 2.0, 0.1, key="svr_c")
            with r2:
                rparams["epsilon"] = st.slider("epsilon", 0.001, 0.5, 0.01, 0.001, key="svr_eps")

        run_pred = st.button("🚀 Train failure prediction model", type="primary")

        if run_pred:
            det = st.session_state.det_result
            y_train = det.loc[train.index, "anomaly_score"]
            y_test = det.loc[test.index, "anomaly_score"]

            with st.spinner("Training regressor..."):
                reg = get_regressor(reg_name, rparams)
                reg.fit(train[feature_columns], y_train)
                pred_train = reg.predict(train[feature_columns])
                pred_test = reg.predict(test[feature_columns])

            y_true_all = pd.concat([y_train, y_test])
            y_pred_all = np.concatenate([pred_train, pred_test])

            mae = mean_absolute_error(y_true_all, y_pred_all)
            rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
            r2s = r2_score(y_true_all, y_pred_all)

            pred_df = pd.DataFrame({
                "actual_score": y_true_all,
                "predicted_score": y_pred_all,
                "split": ["train"] * len(y_train) + ["test"] * len(y_test),
            }, index=y_true_all.index).sort_index()

            st.session_state.pred_result = pred_df
            st.session_state["reg_metrics"] = {"MAE": mae, "RMSE": rmse, "R2": r2s}
            st.session_state["reg_model_name"] = reg_name
            st.session_state["reg_model_obj"] = reg
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

            fig_scatter = px.scatter(pred_df, x="actual_score", y="predicted_score", color="split", title="Predicted vs Actual", opacity=0.6)
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
# TAB 3 — EXPORT
# ==========================================================================
with tab3:
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
