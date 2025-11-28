from pathlib import Path
from typing import Union

import pandas as pd

from .config import DATA_PATH, TARGET_COL


PathLike = Union[str, Path]


def load_weather_data(path: PathLike | None = None) -> pd.DataFrame:
    """
    Load the raw weather CSV as a DataFrame.
    """
    final_path = Path(path) if path is not None else DATA_PATH
    return pd.read_csv(final_path)


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Drop rows where target (RainTomorrow) is missing.
    - Map Yes/No for RainTomorrow and RainToday to 1/0.
    - Optionally drop clearly useless columns (if you want).
    """
    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Drop rows with missing target
    if TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL])

    # Map target Yes/No to 1/0
    if TARGET_COL in df.columns and df[TARGET_COL].dtype == "O":
        df.loc[:, TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    # Map RainToday Yes/No to 1/0 if present
    if "RainToday" in df.columns and df["RainToday"].dtype == "O":
        df.loc[:, "RainToday"] = df["RainToday"].map({"Yes": 1, "No": 0})

    # Example: drop columns that are obviously identifiers or redundant
    for col in ["Date", "Location"]:
        if col in df.columns:
            # you can choose to keep these if you want them as features
            pass

    return df