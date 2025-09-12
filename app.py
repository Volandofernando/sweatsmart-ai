# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import yaml, io
from utils import load_config, load_datasets, detect_features_and_target
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# ---------------- CONFIG ----------------
cfg = load_config("config.yaml")
st.set_page_config(page_title=cfg["app"]["title"], layout="wide")

# ---------------- LOAD DATA ----------------
df = load_datasets(cfg["data"]["path"])
features, target = detect_features_and_target(df, cfg["data"]["target"])

# ---------------- FEATURE ENGINEERING ----------------
def build_feature_vector(temp, humidity, sweat, activity):
    sweat_map = {"Low": 1, "Medium": 2, "High": 3}
    sweat_num = sweat_map[sweat]
    activity_map = {"Low": 1, "Medium": 2, "High": 3}
    activity_num = activity_map[activity]

    return np.array([[
        sweat_num * 5,                 # sweat sensitivity
        800 + humidity * 5,            # absorption baseline
        60 + activity_num * 10,        # ventilation proxy
        0.04 + (temp - 25) * 0.001     # thermal adjustment
    ]])

# ---------------- TRAIN MODEL ----------------
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=cfg["model"]["test_size"], random_state=cfg["model"]["random_state"]
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(**cfg["model"]["params"])
model.fit(X_train_scaled, y_train)

# ---------------- APP UI ----------------
st.title("👕 Fabric Comfort AI Recommender")

tab1, tab2, tab3, tab4 = st.tabs(["Recommender", "Insights", "Model Metrics", "Reports"])

# ---------------- TAB 1: RECOMMENDER ----------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1: temp = st.slider("🌡️ Temperature (°C)", 10, 45, 25)
    with col2: humidity = st.slider("💧 Humidity (%)", 20, 100, 60)
    with col3: sweat = st.selectbox("💦 Sweat Sensitivity", ["Low", "Medium", "High"])
    with col4: activity = st.selectbox("🏃 Activity Level", ["Low", "Medium", "High"])

    if st.button("Get Recommendations"):
        # Predict comfort score
        user_features = build_feature_vector(temp, humidity, sweat, activity)
        user_scaled = scaler.transform(user_features)
        predicted_score = model.predict(user_scaled)[0]

        df["predicted_diff"] = abs(df[target] - predicted_score)
        df["inv_prox"] = 1.0 / (df["predicted_diff"] + 1e-6)
        df["rank_score"] = df["inv_prox"]  # extend with sustainability if available
        df["similarity"] = (df["rank_score"] / df["rank_score"].max() * 100).round(1)

        recommendations = df.sort_values("rank_score", ascending=False).head(5)

        st.subheader("✅ Recommended Fabrics")
        for _, row in recommendations.iterrows():
            st.markdown(
                f"**{row['Fabric']}** – Comfort Score: {row[target]:.2f}, "
                f"Similarity: {row['similarity']}%"
            )

        # Explainability (z-scores)
        zscores = (recommendations[features] - df[features].mean()) / df[features].std()
        st.markdown("### 🔍 Why These Fabrics?")
        for i, row in zscores.iterrows():
            expl = ", ".join([f"{f} {row[f]:+.1f}σ" for f in features])
            st.info(f"{recommendations.loc[i, 'Fabric']}: {expl}")

# ---------------- TAB 2: INSIGHTS ----------------
with tab2:
    st.subheader("📊 Dataset Insights")
    st.write(df.describe())
    corr = df[features + [target]].corr()
    chart = alt.Chart(corr.reset_index()).mark_rect().encode(
        x="index", y="index", color="value"
    )
    st.altair_chart(chart, use_container_width=True)

# ---------------- TAB 3: MODEL METRICS ----------------
with tab3:
    preds = model.predict(X_test_scaled)
    st.metric("R²", f"{r2_score(y_test, preds):.2f}")
    st.metric("RMSE", f"{mean_squared_error(y_test, preds, squared=False):.2f}")
    st.metric("MAE", f"{mean_absolute_error(y_test, preds):.2f}")

# ---------------- TAB 4: REPORTS ----------------
with tab4:
    st.subheader("📥 Download Reports")

    # Excel export
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False)
    st.download_button(
        "⬇️ Full Dataset (Excel)", excel_buffer, "fabric_dataset.xlsx",
        "application/vnd.ms-excel"
    )

    # PDF export
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 800, "Fabric Comfort Recommendation Report")
    c.setFont("Helvetica", 10)
    y = 770
    for _, row in recommendations.iterrows():
        c.drawString(80, y, f"{row['Fabric']} – Comfort: {row[target]:.2f}, Similarity: {row['similarity']}%")
        y -= 20
    c.save()
    st.download_button(
        "⬇️ Recommendations (PDF)", pdf_buffer.getvalue(),
        "fabric_report.pdf", "application/pdf"
    )
