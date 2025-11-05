import numpy as np
import pandas as pd

def compute_comfort_index(model, scaler, parameters, df_clean, target_col):
    """
    Industry-level reusable function to compute comfort score and rank fabrics.
    This removes logic from the UI so results are reproducible + testable.
    """
    # Scale input features
    scaled_inputs = scaler.transform(np.array([parameters]))

    # Model predicts base comfort value
    predicted_score = float(model.predict(scaled_inputs)[0])

    # Normalize prediction to 0–100
    min_val = df_clean[target_col].min()
    max_val = df_clean[target_col].max()
    comfort_index = np.clip(((predicted_score - min_val) / (max_val - min_val)) * 100, 0, 100)

    # Rank fabrics by similarity in comfort behavior
    df_temp = df_clean.copy()
    df_temp["distance"] = abs(df_temp[target_col] - predicted_score)
    recommendations = df_temp.sort_values("distance").groupby("fabric_type")[target_col].mean().reset_index()
    recommendations["comfort_normalized"] = np.clip(((recommendations[target_col] - min_val) / (max_val - min_val)) * 100, 0, 100)
    top3 = recommendations.sort_values("comfort_normalized", ascending=False).head(3)

    return comfort_index, top3
