# db.py
import pandas as pd
from datetime import datetime
import os

LOG_PATH = "user_logs.csv"

def save_user_feedback(input_dict, recommendation):
    """
    input_dict: dictionary of user inputs (values)
    recommendation: recommended fabric / score
    """
    row = dict(input_dict)
    row["recommendation"] = recommendation
    row["timestamp"] = datetime.utcnow().isoformat()
    df_row = pd.DataFrame([row])

    if os.path.exists(LOG_PATH):
        df_row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_PATH, mode="w", header=True, index=False)
