# src/train.py
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import TARGET_COL, TEST_SIZE, RANDOM_STATE, MODEL_PATH, FEATURE_COLS
from .data import load_weather_data, clean_weather_data
from .features import build_pipeline


def train_and_evaluate():
    df = load_weather_data()
    df = clean_weather_data(df)

    # ONLY the columns you want to use
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int)

    pipe, num_cols, cat_cols = build_pipeline(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()