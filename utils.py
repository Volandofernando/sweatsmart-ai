# utils.py
import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Load config
# -------------------------------
def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -------------------------------
# Safe Excel loader (supports GitHub raw URLs)
# -------------------------------
def _read_excel_safe(path_or_url):
    try:
        # use engine openpyxl for xlsx (explicit)
        return pd.read_excel(path_or_url, engine="openpyxl")
    except Exception as e:
        # provide a helpful message
        raise RuntimeError(f"Failed to read Excel file at {path_or_url!r}: {e}")

# -------------------------------
# Load & merge datasets
# -------------------------------
def load_datasets(cfg):
    """
    Expects cfg["datasets"] to contain:
      - material_url (literature)
      - survey_url (responses)
    Returns a concatenated DataFrame (rows from both)
    """
    ds = cfg.get("datasets", {})
    mat_url = ds.get("material_url")
    survey_url = ds.get("survey_url")

    if not mat_url or not survey_url:
        raise KeyError("Config missing dataset URLs: 'material_url' and/or 'survey_url'")

    df_list = []
    # load each and add a source column
    for name, url in [("literature", mat_url), ("survey", survey_url)]:
        df = _read_excel_safe(url)
        df = df.copy()
        df["__source"] = name
        df_list.append(df)

    # concatenate (ignore index, preserve columns union)
    df = pd.concat(df_list, ignore_index=True, sort=False)
    # cleanup column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^\w]", "_", regex=True)
    return df

# -------------------------------
# Find column helper (keywords)
# -------------------------------
def _find_column_by_keywords(cols, keywords):
    """
    Return the first column name that contains ALL tokens in keywords (case-insensitive).
    keywords may be list of strings.
    """
    kws = [k.lower() for k in keywords]
    for c in cols:
        cl = c.lower()
        if all(k in cl for k in kws):
            return c
    return None

# -------------------------------
# Detect features & target intelligently
# -------------------------------
def detect_features_and_target(df, cfg):
    """
    Use config keywords to find the 4 required features and the target.
    Config example:
      feature_keywords:
        moisture_regain: ["moisture", "regain"]
        water_absorption: ["water", "absorption"]
        drying_time: ["drying", "time"]
        thermal_conductivity: ["thermal", "conductivity"]
      target_keywords: ["comfort", "score"]
    Returns (feature_cols_list, target_col_name)
    """
    cols = list(df.columns)

    fk = cfg.get("feature_keywords", {})
    tk = cfg.get("target_keywords", ["comfort", "score"])

    # try configured keywords first
    feature_cols = []
    for k, keywords in fk.items():
        found = _find_column_by_keywords(cols, keywords)
        if found:
            feature_cols.append(found)

    # find target
    target_col = _find_column_by_keywords(cols, tk)

    # If we didn't find all 4 features, try to fall back to sensible numeric columns:
    if len(feature_cols) < 4:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # remove target if detected
        if target_col in numeric_cols:
            numeric_cols = [c for c in numeric_cols if c != target_col]
        # append missing from numeric list (prefer order)
        for c in numeric_cols:
            if c not in feature_cols:
                feature_cols.append(c)
            if len(feature_cols) >= 4:
                break

    # final validation
    feature_cols = [c for c in feature_cols if c in cols]
    if target_col is None:
        # fallback: last numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            target_col = numeric_cols[-1]
        else:
            target_col = None

    # ensure exactly 4 features (or at least 1) — caller expects >=4, but handle gracefully
    return feature_cols, target_col

# -------------------------------
# Train model (returns model, scaler, X_test, y_test, df_clean)
# -------------------------------
def train_model(df, feature_cols, target_col, cfg):
    if not feature_cols or target_col is None:
        raise ValueError("train_model requires feature_cols and target_col")

    # drop rows missing required data
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()

    if df_clean.shape[0] < 10:
        # warn but continue
        # still attempt training if there are at least 3 rows
        if df_clean.shape[0] < 3:
            raise ValueError("Not enough rows after dropping NA to train a model (need >=3).")

    X = df_clean[feature_cols].astype(float).values
    y = df_clean[target_col].astype(float).values

    test_size = float(cfg.get("model", {}).get("test_size", 0.2))
    random_state = int(cfg.get("model", {}).get("random_state", 42))
    n_estimators = int(cfg.get("model", {}).get("n_estimators", 200))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test_scaled, y_test, df_clean

# -------------------------------
# Evaluate model
# -------------------------------
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    return {"rmse": round(rmse, 4), "r2": round(r2, 4)}
