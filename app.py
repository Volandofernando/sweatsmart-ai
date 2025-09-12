import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils import load_config, load_datasets, detect_features_and_target, train_model, evaluate_model

# -------------------------------
# Load Config
# -------------------------------
cfg = load_config()
st.set_page_config(page_title=cfg["app"]["title"], layout="wide")

# -------------------------------
# Branding
# -------------------------------
st.title(f"👕 {cfg['app']['title']}")
st.subheader("Comfort & Performance Insights for Apparel Industry")

# -------------------------------
# Load Data
# -------------------------------
try:
    df_material, df_survey, df = load_datasets(cfg)
except Exception as e:
    st.error(f"❌ Failed to load datasets: {e}")
    st.stop()

features, target = detect_features_and_target(df, cfg)
if target is None:
    st.error("❌ Dataset does not have required features or target.")
    st.stop()

model, scaler, X_test, y_test, df_clean = train_model(df, features, target, cfg)

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📌 Recommender", "📊 Insights", "🤖 Model Performance", "ℹ️ About"]
)

# -------------------------------
# TAB 1: Recommender
# -------------------------------
with tab1:
    st.markdown("### ⚙️ Set Environment Conditions")
    temp = st.slider("🌡️ Outdoor Temperature (°C)", 10, 45, 28)
    humidity = st.slider("💧 Humidity (%)", 10, 100, 60)
    sweat = st.select_slider("🧍 Sweat Sensitivity", ["Low", "Medium", "High"])
    activity = st.select_slider("🏃 Activity Intensity", ["Low", "Moderate", "High"])

    sweat_map = {"Low": 1, "Medium": 2, "High": 3}
    act_map = {"Low": 1, "Moderate": 2, "High": 3}

    user_input = np.array([[sweat_map[sweat] * 5,
                            800 + humidity * 5,
                            60 + act_map[activity] * 10,
                            0.04 + (temp - 25) * 0.001]])
    user_scaled = scaler.transform(user_input)
    score = model.predict(user_scaled)[0]

    df_clean["predicted_diff"] = abs(df_clean[target] - score)
    top_matches = df_clean.sort_values("predicted_diff").head(3)

    st.markdown("## 🔹 Recommended Fabrics")
    for _, row in top_matches.iterrows():
        st.metric(label=f"🧵 {row.get('fabric_type','Unknown')}",
                  value=round(row[target], 2),
                  delta="Comfort Score")

    st.download_button(
        "⬇️ Download Recommendations (CSV)",
        top_matches.to_csv(index=False),
        "recommendations.csv",
        "text/csv"
    )

# -------------------------------
# TAB 2: Insights
# -------------------------------
with tab2:
    st.markdown("### 📊 Material Dataset")
    st.dataframe(df_material.head(10))
    st.download_button("⬇️ Download Material Data", df_material.to_csv(index=False), "material.csv")

    st.markdown("### 📊 Survey Dataset")
    st.dataframe(df_survey.head(10))
    st.download_button("⬇️ Download Survey Data", df_survey.to_csv(index=False), "survey.csv")

    st.markdown("### 📊 Combined Training Dataset")
    st.dataframe(df_clean.head(10))

# -------------------------------
# TAB 3: Model Performance
# -------------------------------
with tab3:
    metrics = evaluate_model(model, X_test, y_test)
    st.metric("R² Score", metrics["r2"])
    st.metric("RMSE", metrics["rmse"])

    st.write("#### Feature Importances")
    feat_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
    chart = alt.Chart(feat_df).mark_bar(color=cfg["app"]["theme_color"]).encode(x="Feature", y="Importance")
    st.altair_chart(chart, use_container_width=True)

# -------------------------------
# TAB 4: About
# -------------------------------
with tab4:
    st.markdown(f"""
    **{cfg['app']['title']}**  
    A professional AI system for **fabric comfort recommendation**.  

    🚀 Features:  
    - AI-powered predictions  
    - Industry datasets (material + survey)  
    - Downloadable reports  
    - Interactive visual insights  
    """)
