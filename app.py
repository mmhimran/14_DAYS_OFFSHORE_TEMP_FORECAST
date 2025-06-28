import streamlit as st
import pandas as pd
import numpy as np
import io
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="14-DAY OFFSHORE TEMP FORECAST", layout="wide", initial_sidebar_state="expanded")

# ------------------- CSS STYLING -------------------
st.markdown("""
    <style>
        .main, .block-container {
            background-color: #417C7B !important;
        }
        .css-1d391kg, .css-1cpxqw2 {
            color: #FF0000 !important;
            font-size: 18px !important;
        }
        .css-10trblm {
            color: #FFFAFA !important;
            font-size: 26px !important;
            font-weight: bold !important;
        }
        h1, h2, h3 {
            color: #FFFAFA !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("hybrid_lstm_all.h5")
model = load_model()

# ------------------- FORECAST FUNCTION -------------------
def predict_and_boost(df_input, lookback=1008, forecast_horizon=336):
    df_input = df_input.fillna(method='ffill')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_input[['Layer 1', 'Layer 2', 'Layer 3']])
    X_input = np.array([scaled[-lookback:]])

    # Predict with LSTM
    lstm_pred_scaled = model.predict(X_input).reshape(forecast_horizon, 3)

    # Apply XGBoost for each layer
    boosted_pred_scaled = np.zeros_like(lstm_pred_scaled)
    true_scaled = scaled[lookback:lookback + forecast_horizon]
    for i in range(3):
        X_gb = [[lstm_pred_scaled[t][i], t / forecast_horizon] for t in range(forecast_horizon)]
        y_gb = true_scaled[:, i] if len(true_scaled) == forecast_horizon else lstm_pred_scaled[:, i]
        booster = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1)
        booster.fit(X_gb, y_gb)
        boosted_pred_scaled[:, i] = booster.predict(X_gb)

    # Inverse scale
    padded = np.zeros((forecast_horizon, 3))
    padded[:, :] = boosted_pred_scaled
    forecast = scaler.inverse_transform(padded)

    forecast_dates = pd.date_range(start=df_input['Date'].iloc[-1] + timedelta(hours=1), periods=forecast_horizon, freq='H')
    forecast_df = pd.DataFrame(forecast, columns=['Layer 1', 'Layer 2', 'Layer 3'])
    forecast_df.insert(0, 'Date', forecast_dates)
    return forecast_df

# ------------------- COMPARE FUNCTION -------------------
def compare_forecast(actual_df, predicted_df):
    merged = pd.merge(actual_df, predicted_df, on='Date')
    result = pd.DataFrame({'Date': merged['Date']})
    for col in ['Layer 1', 'Layer 2', 'Layer 3']:
        result[f'Actual_{col}'] = merged[col]
        result[f'Predicted_{col}'] = merged[f'{col}']
        result[f'Error_{col}'] = result[f'Actual_{col}'] - result[f'Predicted_{col}']
        result[f'Accuracy_{col}'] = 100 - (np.abs(result[f'Error_{col}']) / result[f'Actual_{col}'] * 100)
    return result

# ------------------- UTILITIES -------------------
def generate_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast')
    return output.getvalue()

def plot_colored_line(df, x, y, title):
    fig = px.line(df, x=x, y=y,
                  labels={'value': 'Temperature (\u00b0C)', 'variable': 'Legend'},
                  title=title,
                  color_discrete_sequence=['blue', 'red'])
    fig.update_traces(line=dict(width=4), hovertemplate='<b>%{y:.2f}</b>', hoverlabel=dict(font_color='red'))
    fig.update_layout(
        font=dict(family="Times New Roman", size=24, color="black"),
        title_font=dict(size=28, family="Times New Roman", color="black"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, tickfont=dict(size=20, color='black'), title_font=dict(size=24)),
        yaxis=dict(showgrid=True, tickfont=dict(size=20, color='black'), title_font=dict(size=24)),
        margin=dict(l=50, r=50, t=80, b=50),
        hoverlabel=dict(bgcolor="white", font_size=20, font_family="Times New Roman")
    )
    return fig

def round_df(df, decimals=2):
    return df.round({col: decimals for col in df.columns if col != 'Date'})

# ------------------- STREAMLIT UI -------------------
st.title("\U0001F9F0 14-Day Offshore Temperature Forecast Dashboard")
mode = st.sidebar.radio("Choose Mode", [
    "Forecast Only",
    "Forecast + Actual Comparison",
    "Visualize Actual vs Predicted",
    "Visualize Accuracy",
    "Download Forecast"
])

# ------------------- MAIN BLOCKS -------------------

if mode == "Forecast Only":
    file = st.file_uploader("Upload input file with 1008 hourly records", type=['xlsx', 'csv'])
    if file:
        df = pd.read_excel(file) if file.name.endswith('xlsx') else pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        forecast_df = predict_and_boost(df)
        st.dataframe(forecast_df.head())
        st.download_button("\U0001F4E5 Download Forecast Excel", generate_excel(forecast_df), file_name="14day_forecast.xlsx")

elif mode == "Forecast + Actual Comparison":
    file = st.file_uploader("Upload Excel file with at least 1344 records (1008+336)", type=['xlsx'])
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df_input = df.iloc[:1008]
        df_actual = df.iloc[1008:1344].reset_index(drop=True)
        forecast_df = predict_and_boost(df_input)
        forecast_df.columns = ['Date', 'Layer 1', 'Layer 2', 'Layer 3']
        comparison = compare_forecast(df_actual, forecast_df)
        st.dataframe(comparison.head())
        st.download_button("\U0001F4E5 Download Comparison Excel", generate_excel(comparison), file_name="14day_comparison.xlsx")

elif mode == "Visualize Actual vs Predicted":
    file = st.file_uploader("Upload result file with 'Actual' and 'Predicted' columns", type=['xlsx'])
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Layer 1', 'Layer 2', 'Layer 3']:
            if f'Actual_{col}' in df.columns and f'Predicted_{col}' in df.columns:
                fig = plot_colored_line(df, x='Date', y=[f'Actual_{col}', f'Predicted_{col}'],
                                        title=f"Actual vs Predicted - {col}")
                st.plotly_chart(fig, use_container_width=True)

elif mode == "Visualize Accuracy":
    file = st.file_uploader("Upload result file with accuracy columns", type=['xlsx'])
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Layer 1', 'Layer 2', 'Layer 3']:
            if f'Accuracy_{col}' in df.columns:
                fig = plot_colored_line(df, x='Date', y=f'Accuracy_{col}', title=f"Accuracy - {col}")
                st.plotly_chart(fig, use_container_width=True)

elif mode == "Download Forecast":
    file = st.file_uploader("Upload file for forecast only", type=['xlsx', 'csv'])
    if file:
        df = pd.read_excel(file) if file.name.endswith('xlsx') else pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        forecast_df = predict_and_boost(df)
        st.download_button("\U0001F4E5 Download Forecast Excel", generate_excel(forecast_df), file_name="14day_forecast_only.xlsx")
