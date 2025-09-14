import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import math

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    url1 = "https://github.com/Volandofernando/Material-Literature-data-/raw/main/Dataset.xlsx"
    url2 = "https://github.com/Volandofernando/REAL-TIME-Dataset/raw/main/IT%20Innovation%20in%20Fabric%20Industry%20%20(Responses).xlsx"
    df1 = pd.read_excel(url1)
    df2 = pd.read_excel(url2)
    return pd.concat([df1, df2], axis=0, ignore_index=True)

df = load_data()

# -------------------------------
# Train Model
# -------------------------------
features = ["Temperature", "Humidity", "Activity_Level"]
target = "Sweat_Rate"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions & Metrics
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="SweatSmart AI Recommender", layout="wide")

st.title("👕 Fabric Comfort AI Recommender")
st.subheader("AI-powered comfort & sweat management insights for apparel innovation")

# Sidebar Guidance
with st.sidebar:
    st.header("ℹ️ How Inputs Work")
    st.markdown("""
    - **Temperature (°C):** Warmer conditions increase sweat.
    - **Humidity (%):** High humidity reduces sweat evaporation.
    - **Activity Level (1-10):** Higher = more intense activity.
    - The AI model predicts sweat rate → recommends fabrics.
    """)

# User Inputs
temp = st.slider("🌡️ Temperature (°C)", 15, 45, 30)
hum = st.slider("💧 Humidity (%)", 10, 100, 60)
activity = st.slider("🏃 Activity Level", 1, 10, 5)

# Prediction
pred_sweat = model.predict([[temp, hum, activity]])[0]
pred_percent = min(max((pred_sweat / y.max()) * 100, 0), 100)

st.metric("Predicted Sweat Rate", f"{pred_sweat:.2f} L/hr", f"{pred_percent:.1f}% of max observed")

# -------------------------------
# Recommendations
# -------------------------------
st.subheader("🎯 Recommended Fabrics for Your Conditions")

recommendations = {
    "Polyester": "Durable, lightweight, dries fast. Often used in sportswear.",
    "Cotton": "Soft & breathable, but absorbs sweat and dries slowly.",
    "Nylon": "Strong, lightweight, quick-drying, but less breathable.",
    "Merino Wool": "Regulates temperature, resists odor, good for varied climates.",
    "Bamboo Fabric": "Eco-friendly, breathable, antibacterial properties."
}

rec_df = pd.DataFrame(recommendations.items(), columns=["Fabric", "Description"])
st.table(rec_df)

# Download Option
st.download_button(
    label="📥 Download Recommendation Report (Excel)",
    data=rec_df.to_csv(index=False),
    file_name="fabric_recommendations.csv",
    mime="text/csv"
)

# -------------------------------
# Model Performance Explanation
# -------------------------------
with st.expander("📊 What do R² and RMSE mean?"):
    st.write(f"**R² Score:** {r2:.2f} ({r2*100:.1f}% accuracy in predicting sweat rates)")
    st.write(f"**RMSE:** {rmse:.2f} → On average, predictions are within ±{rmse:.2f} L/hr of true values")

    st.markdown("""
    **Beginner-Friendly Guide:**
    - **R² (Coefficient of Determination):** Measures how well the model explains sweat rate.  
      - 0.90 = *Excellent (90% accurate)*  
      - 0.50 = *Moderate*  
      - 0.20 = *Weak*  
    - **RMSE (Root Mean Square Error):** Measures average prediction error.  
      - Lower = Better (closer to real-world values).  
    """)

# -------------------------------
# User Guidance
# -------------------------------
with st.expander("🧭 Understanding Fabric Comfort"):
    st.markdown("""
    - **Humidity:** The higher it is, the harder sweat evaporates → you feel stickier.
    - **Temperature:** Hotter weather = more sweat production.
    - **Activity:** Running generates more sweat than walking.
    - This AI combines these to recommend fabrics balancing breathability & moisture management.
    """)
