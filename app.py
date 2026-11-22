import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
st.set_page_config(
    page_title="WindCast SCADA | Forecasting & Fault Detection",
    page_icon="🌬️",
    layout="wide"
)
st.title("🌬️ WindCast SCADA: Wind Turbine Power Generation Forecasting & Fault Detection")
st.markdown("SCADA Telemetry Analysis • Multi-Method Feature Selection • Random Forest Ensemble • Residual Anomaly Detection")
@st.cache_resource
def run_pipeline(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("T1.csv")
    df.columns = df.columns.str.strip()
    rename_dict = {
        "Date/Time": "Timestamp",
        "LV ActivePower (kW)": "ActivePower",
        "Wind Speed (m/s)": "WindSpeed",
        "Theoretical_Power_Curve (KWh)": "TheoreticalPower",
        "Wind Direction (°)": "WindDirection",
    }
    df = df.rename(columns=rename_dict)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d %m %Y %H:%M")
    df["Hour"] = df["Timestamp"].dt.hour
    df["Month"] = df["Timestamp"].dt.month
    initial_len = len(df)
    df = df.dropna()
    df_clean = df[(df["ActivePower"] >= 0) & (df["WindSpeed"] >= 0)].copy()
    df_clean["WindDir_Sin"] = np.sin(np.radians(df_clean["WindDirection"]))
    df_clean["WindDir_Cos"] = np.cos(np.radians(df_clean["WindDirection"]))
    candidate_features = [
        "WindSpeed", "TheoreticalPower", "WindDirection",
        "WindDir_Sin", "WindDir_Cos", "Hour", "Month"
    ]
    target = "ActivePower"
    X = df_clean[candidate_features]
    y = df_clean[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    K_FEATURES = 3
    feature_names = np.array(candidate_features)
    filter_sel = SelectKBest(score_func=f_regression, k=K_FEATURES).fit(X_train_scaled, y_train)
    filter_feats = feature_names[filter_sel.get_support()].tolist()
    rfe_sel = RFE(estimator=LinearRegression(), n_features_to_select=K_FEATURES).fit(X_train_scaled, y_train)
    rfe_feats = feature_names[rfe_sel.get_support()].tolist()

    rf_base = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_base.fit(X_train_scaled, y_train)
    importances = rf_base.feature_importances_
    embedded_feats = feature_names[np.argsort(importances)[::-1][:K_FEATURES]].tolist()

    def eval_subset(cols, label):
        idx = [candidate_features.index(c) for c in cols]
        model = RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled[:, idx], y_train)
        preds = model.predict(X_test_scaled[:, idx])
        return {
            "Subset": label,
            "Num_Features": len(cols),
            "Features": ", ".join(cols),
            "R2_Score": r2_score(y_test, preds),
            "RMSE (kW)": np.sqrt(mean_squared_error(y_test, preds)),
            "MAE (kW)": mean_absolute_error(y_test, preds),
            "Model": model,
            "Predictions": preds,
            "Indices": idx
        }
    res_all=eval_subset(candidate_features, "All Features (Baseline)")
    res_filter=eval_subset(filter_feats, "Filter Selection (ANOVA/F-Reg)")
    res_wrapper=eval_subset(rfe_feats, "Wrapper Selection (RFE)")
    res_embedded=eval_subset(embedded_feats, "Embedded Selection (RF MDI)")
    results_df=pd.DataFrame([
        {k: v for k, v in r.items() if k not in ["Model", "Predictions", "Indices"]}
        for r in [res_all, res_filter, res_wrapper, res_embedded]
    ])
    test_actuals=y_test.values
    test_preds=res_embedded["Predictions"]
    test_wind_speeds=X_test["WindSpeed"].values
    residuals=np.abs(test_actuals-test_preds)
    threshold=np.mean(residuals)+2.0*np.std(residuals)
    is_anomaly=residuals>threshold
    return {
        "initial_len": initial_len,
        "clean_len": len(df_clean),
        "candidate_features": candidate_features,
        "importances": importances,
        "results_df": results_df,
        "best_model": res_embedded["Model"],
        "best_feats": embedded_feats,
        "best_idx": res_embedded["Indices"],
        "scaler": scaler,
        "test_actuals": test_actuals,
        "test_wind_speeds": test_wind_speeds,
        "threshold": threshold,
        "is_anomaly": is_anomaly
    }

st.sidebar.header("📂 Dataset Configuration")
uploaded_csv = st.sidebar.file_uploader("Upload SCADA CSV (Optional)", type=["csv"])

try:
    data = run_pipeline(uploaded_csv)
except FileNotFoundError:
    st.error("⚠️ `T1.csv` not found in the repository. Please upload `T1.csv` in the sidebar!")
    st.stop()
st.sidebar.divider()
st.sidebar.header("🎛️ Live Fault Simulator")
st.sidebar.caption("Simulate live turbine sensor inputs to test the Random Forest model & anomaly alarm.")
sim_wind=st.sidebar.slider("Wind Speed (m/s)", 0.0, 25.0, 12.0, 0.5)
default_theo=min(3600.0, max(0.0, 3.2 * (sim_wind ** 3))) if sim_wind >= 3.5 else 0.0
sim_theo = st.sidebar.number_input("Theoretical Power (kW)", 0.0, 3600.0, float(round(default_theo, 1)))
sim_dir = st.sidebar.slider("Wind Direction (°)", 0, 360, 180)
sim_actual = st.sidebar.number_input("Measured Active Power (kW)", 0.0, 3600.0, float(round(default_theo * 0.95, 1)))
dummy_row = np.zeros((1, 7))
dummy_row[0, 0] = sim_wind
dummy_row[0, 1] = sim_theo
dummy_row[0, 2] = sim_dir
scaled_row = data["scaler"].transform(dummy_row)
pred_power = data["best_model"].predict(scaled_row[:, data["best_idx"]])[0]
live_residual = abs(sim_actual - pred_power)
is_fault = live_residual > data["threshold"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clean SCADA Records", f"{data['clean_len']:,}", f"Dropped {data['initial_len'] - data['clean_len']} invalid")
c2.metric("Selected Features (Embedded)", "3 / 7", ", ".join(data["best_feats"]))
c3.metric("Dynamic Anomaly Threshold", f"{data['threshold']:.2f} kW", "μ + 2σ Residual Cutoff")
c4.metric("Detected Faults (Test Set)", f"{np.sum(data['is_anomaly'])} pts", f"{np.mean(data['is_anomaly'])*100:.2f}% of test data")
st.subheader("🔍 Live Turbine Diagnostics")
sim_col1, sim_col2, sim_col3 = st.columns(3)
sim_col1.metric("Predicted Expected Power", f"{pred_power:.2f} kW")
sim_col2.metric("Residual Error (|Actual - Pred|)", f"{live_residual:.2f} kW")
if is_fault:
    sim_col3.error(f"🚨 ANOMALY FLAGGED! Error exceeds {data['threshold']:.2f} kW")
else:
    sim_col3.success("✅ NORMAL OPERATION (Within healthy envelope)")
st.divider()
tab1, tab2 = st.tabs(["📊 Power Curve & Feature Importance", "📋 Feature Selection Benchmark"])
with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.barplot(
        x=data["importances"],
        y=data["candidate_features"],
        palette="Blues_r",
        ax=axes[0]
    )
    axes[0].set_title("Embedded Feature Importance (Random Forest MDI)", fontsize=12)
    axes[0].set_xlabel("Relative Importance Score")

    mask = data["is_anomaly"]
    axes[1].scatter(
        data["test_wind_speeds"][~mask], data["test_actuals"][~mask],
        alpha=0.25, s=12, color="steelblue", label="Normal Generation"
    )
    axes[1].scatter(
        data["test_wind_speeds"][mask], data["test_actuals"][mask],
        alpha=0.75, s=25, color="crimson", label="Underperformance / Anomaly"
    )
    axes[1].scatter([sim_wind], [sim_actual], color="gold", edgecolor="black", s=180, marker="*", zorder=5, label="Live Simulator Point")
    axes[1].set_title("Turbine Power Curve with Flagged Operating Anomalies", fontsize=12)
    axes[1].set_xlabel("Wind Speed (m/s)")
    axes[1].set_ylabel("LV Active Power (kW)")
    axes[1].legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)
with tab2:
    st.markdown("### Feature Selection & Regression Performance Comparison")
    st.dataframe(
        data["results_df"].style.highlight_max(subset=["R2_Score"], color="#d4edda")
                                .highlight_min(subset=["RMSE (kW)", "MAE (kW)"], color="#d4edda"),
        use_container_width=True
    )
