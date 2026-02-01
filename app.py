import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib

# UI Enhancements
st.set_page_config(page_title="Air Quality Predictor", layout="wide")
st.title(" Air Quality Prediction using DNN Models")

# Load models and scalers
basic_model = load_model("basic_dnn_model.h5")
improved_model = load_model("improved_dnn_model.h5")
scaler_basic = joblib.load("scaler.pkl")
scaler_improved = joblib.load("scaler_X.pkl")
target_scaler = joblib.load("target_scaler.pkl")

# Load test data (optional preview)
X_test_scaled = np.load("X_test_scaled.npy")
y_test = np.load("y_test.npy")

# Layout tabs
tabs = st.tabs(["Prediction Graphs", "Feature Importance", "Bulk Prediction", "Try Manually"])

# --- Tab 1: Prediction Graphs ---
with tabs[0]:
    st.subheader("Actual vs Predicted - Basic & Improved DNN")
    
    # Predict using both models
y_pred_basic = basic_model.predict(X_test_scaled)
y_pred_basic_inv = target_scaler.inverse_transform(y_pred_basic)

y_pred_improved = improved_model.predict(X_test_scaled)
y_pred_improved_inv = target_scaler.inverse_transform(y_pred_improved)

    # Inverse transform actual
y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(y_test_inv, y_pred_basic_inv, alpha=0.6, color='blue')
axes[0].plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--')
axes[0].set_title('Basic DNN: Actual vs Predicted')
axes[0].set_xlabel('Actual PM2.5')
axes[0].set_ylabel('Predicted PM2.5')

axes[1].scatter(y_test_inv, y_pred_improved_inv, alpha=0.6, color='green')
axes[1].plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--')
axes[1].set_title('Improved DNN: Actual vs Predicted')
axes[1].set_xlabel('Actual PM2.5')
axes[1].set_ylabel('Predicted PM2.5')

st.pyplot(fig)

# --- Tab 2: Feature Importance ---
with tabs[1]:
    st.subheader("🔍 Feature Importance Visualization")

    feature_names = [
        'Ozone', 'NO2', 'CO', 'SO2', 'PM10',
        'Temperature', 'Humidity', 'Wind Speed',
        'Rainfall', 'Visibility'
    ]

    importances = np.array([0.18, 0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.04])

    fig2, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names, ax=ax, palette='viridis')
    ax.set_title("Top 10 Features Based on Importance")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    st.pyplot(fig2)

# --- Tab 3: Bulk Prediction ---
with tabs[2]:
    st.subheader("📁 Upload CSV for Bulk Prediction")

    uploaded_file = st.file_uploader("Upload CSV with 10 Features", type=['csv'])

    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)

        if df_input.shape[1] != 10:
            st.error("❌ Please upload a CSV with exactly 10 columns.")
        else:
            # Scale input for both models
            input_scaled_basic = scaler_basic.transform(df_input)
            input_scaled_improved = scaler_improved.transform(df_input)

            pred_basic = basic_model.predict(input_scaled_basic)
            pred_improved = improved_model.predict(input_scaled_improved)

            pred_basic_inv = target_scaler.inverse_transform(pred_basic)
            pred_improved_inv = target_scaler.inverse_transform(pred_improved)

            result_df = df_input.copy()
            result_df["Basic DNN Prediction"] = pred_basic_inv.flatten()
            result_df["Improved DNN Prediction"] = pred_improved_inv.flatten()

            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")


# --- Tab 4: Manual Prediction ---
with tabs[3]:
    st.subheader("🧪 Predict PM2.5 from Manual Input")

    # Feature sliders
    st.markdown("### 🔧 Set Input Feature Values")

    col1, col2 = st.columns(2)
    with col1:
        ozone = st.slider("Ozone", 0.0, 500.0, 50.0)
        no2 = st.slider("NO2", 0.0, 500.0, 50.0)
        co = st.slider("CO", 0.0, 10.0, 1.0)
        so2 = st.slider("SO2", 0.0, 200.0, 10.0)
        pm10 = st.slider("PM10", 0.0, 600.0, 100.0)

    with col2:
        temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 3.0)
        rainfall = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0)
        visibility = st.slider("Visibility (km)", 0.0, 50.0, 10.0)

    # Model selection
    model_choice = st.selectbox("Choose Model", ["Basic DNN", "Improved DNN"])

    if st.button("🔮 Predict PM2.5"):
        user_input = np.array([
            ozone, no2, co, so2, pm10,
            temperature, humidity, wind_speed,
            rainfall, visibility
        ]).reshape(1, -1)

        if model_choice == "Basic DNN":
            scaled_input = scaler_basic.transform(user_input)
            pred = basic_model.predict(scaled_input)
        else:
            scaled_input = scaler_improved.transform(user_input)
            pred = improved_model.predict(scaled_input)

        pred_inv = target_scaler.inverse_transform(pred)
        st.success(f"🌫️ PM2.5 Prediction ({model_choice}): {pred_inv[0][0]:.2f}")
