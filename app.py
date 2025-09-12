# utils.py
import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _read_excel(url_or_path):
    # explicit engine for .xlsx files
    return pd.read_excel(url_or_path, engine="openpyxl")

def load_datasets(cfg):
    ds = cfg.get("datasets", {})
    mat = ds.get("material_url")
    survey = ds.get("survey_url")
    if not mat or not survey:
        raise KeyError("config.yaml must have datasets.material_url and datasets.survey_url")
    dm = _read_excel(mat)
    dsr = _read_excel(survey)
    # unify: lower/clean columns, add source
    dm = dm.copy()
    dm["__source"] = "literature"
    dsr = dsr.copy()
    dsr["__source"] = "survey"
    df = pd.concat([dm, dsr], ignore_index=True, sort=False)
    # standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^\w]", "_", regex=True)
    return df

def _find_column_by_keywords(cols, keywords):
    kws = [k.lower() for k in keywords]
    for c in cols:
        cl = c.lower()
        if all(k in cl for k in kws):
            return c
    return None

def detect_features_and_target(df, cfg):
    fk = cfg.get("feature_keywords", {})
    tk = cfg.get("target_keywords", ["comfort", "score"])
    cols = df.columns.tolist()
    feature_cols = []
    for _, kws in fk.items():
        found = _find_column_by_keywords(cols, kws)
        if found:
            feature_cols.append(found)
    target_col = _find_column_by_keywords(cols, tk)
    # fallback: choose numeric columns
    if len(feature_cols) < 4:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # remove target if detected
        if target_col in numeric_cols:
            numeric_cols = [c for c in numeric_cols if c != target_col]
        for c in numeric_cols:
            if c not in feature_cols:
                feature_cols.append(c)
            if len(feature_cols) >= 4:
                break
    if target_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            target_col = numeric_cols[-1]
    return feature_cols, target_col

def train_model(df, feature_cols, target_col, cfg):
    # ensure target numeric
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=feature_cols + [target_col])
    if df.shape[0] < 3:
        raise ValueError("Not enough valid rows to train model after dropna.")
    X = df[feature_cols].astype(float).values
    y = df[target_col].astype(float).values
    test_size = float(cfg.get("model", {}).get("test_size", 0.25))
    rs = int(cfg.get("model", {}).get("random_state", 42))
    n_estimators = int(cfg.get("model", {}).get("n_estimators", 200))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=rs)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X_test_scaled, y_test, df

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    return {"rmse": round(rmse, 4), "r2": round(r2, 4)}
