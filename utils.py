import pandas as pd
import yaml
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# -------------------------------
# Load Config
# -------------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -------------------------------
# Load Datasets
# -------------------------------
def load_datasets(cfg):
    try:
        material = pd.read_excel(cfg["datasets"]["material_data"])
        survey = pd.read_excel(cfg["datasets"]["survey_data"])
    except Exception as e:
        raise ValueError(f"Error loading datasets: {e}")

    # Drop NA
    material = material.dropna()
    survey = survey.dropna()

    # Merge only on common columns
    common_cols = list(set(material.columns).intersection(set(survey.columns)))
    combined = pd.concat([material[common_cols], survey[common_cols]], ignore_index=True)

    return material, survey, combined

# -------------------------------
# Feature + Target Detection
# -------------------------------
def detect_features_and_target(df, cfg):
    features = cfg["ml"]["features"]
    target = cfg["ml"]["target"]
    if not all(f in df.columns for f in features) or target not in df.columns:
        return None, None
    return features, target

# -------------------------------
# Train Model
# -------------------------------
def train_model(df, features, target, cfg):
    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=cfg["ml"]["n_estimators"], random_state=42
    )
    model.fit(X_train, y_train)

    return model, scaler, X_test, y_test, df

# -------------------------------
# Model Evaluation
# -------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "r2": round(r2_score(y_test, y_pred), 3),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
    }
