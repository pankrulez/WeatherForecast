# app.py
import streamlit as st

from src.data import load_weather_data, clean_weather_data
from src.predict import predict_rain

# ---------- Page functions ----------

def show_cleaning_page() -> None:
    st.title("Weather Data – Cleaning & Exploration")

    st.markdown("### 1. Raw data")
    df_raw = load_weather_data()
    st.write(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    st.dataframe(df_raw.head())

    st.markdown("### 2. Missing values")
    missing = df_raw.isna().sum().sort_values(ascending=False)
    st.dataframe(missing.to_frame("missing_count"))

    st.markdown("### 3. Cleaned data (after `clean_weather_data`)")
    df_clean = clean_weather_data(df_raw.copy())
    st.write(f"Shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
    st.dataframe(df_clean.head())

    with st.expander("Show cleaning steps description"):
        st.write(
            """
            This cleaned dataset is produced by `clean_weather_data` in `src/data.py`.
            Modify that function to:
            - Drop rows with critical missing values
            - Encode Yes/No columns to 1/0
            - Drop or impute columns with too many missing values
            - Any other logic you had in your notebook
            """
        )


def show_prediction_page() -> None:
    st.title("Australian Weather – Rain Tomorrow Predictor")

    st.write("Fill in the weather details to predict if it will rain tomorrow.")

    # Replace/add fields to match your actual model features
    min_temp = st.number_input("MinTemp", value=10.0)
    max_temp = st.number_input("MaxTemp", value=20.0)
    humidity_3pm = st.number_input("Humidity3pm", value=60.0)
    pressure_3pm = st.number_input("Pressure3pm", value=1015.0)
    wind_speed_3pm = st.number_input("WindSpeed3pm", value=15.0)
    rain_today = st.selectbox("RainToday", ["No", "Yes"])

    if st.button("Predict"):
        input_data = {
            "MinTemp": min_temp,
            "MaxTemp": max_temp,
            "Humidity3pm": humidity_3pm,
            "Pressure3pm": pressure_3pm,
            "WindSpeed3pm": wind_speed_3pm,
            "RainToday": rain_today,
            # Add all other features the model expects, with sensible defaults
        }

        result = predict_rain(input_data)
        label = "Rain" if result["prediction"] == 1 else "No Rain"

        st.subheader(f"Prediction: {label}")
        st.write(f"Probability of rain: {result['probability']:.2%}")


# ---------- Main app ----------

def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Data cleaning & EDA", "Prediction"],
    )

    if page == "Data cleaning & EDA":
        show_cleaning_page()
    else:
        show_prediction_page()


if __name__ == "__main__":
    main()