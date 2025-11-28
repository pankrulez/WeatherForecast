from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "AUS_Weather.csv"
MODEL_PATH = BASE_DIR / "models" / "weather_logreg.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "RainTomorrow"

FEATURE_COLS = [
    "MinTemp",
    "MaxTemp",
    "Humidity3pm",
    "Pressure3pm",
    "WindSpeed3pm",
    "RainToday",
]
