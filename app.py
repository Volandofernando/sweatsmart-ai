import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="👕 SweatSmart AI – Fabric Comfort Recommender",
    layout="wide"
)

# Fabric explanations (extend as needed)
FABRIC_INFO = {
    "Cotton": "Soft, breathable, and moisture-absorbent. Great for everyday wear, but dries slowly.",
    "Polyester": "Durable, wrinkle-resistant, and lightweight. Wicks moisture but can trap heat.",
    "Nylon": "Strong, elastic, quick-drying, commonly used in activewear.",
    "Wool": "Warm, insulating, and naturally breathable. Can be itchy for sensitive skin.",
    "Linen": "Highly breathable and cool, but wrinkles easily.",
    "Bamboo": "Soft, eco-friendly, and breathable. Naturally antibacterial."
}

# ----------------------------
# LOAD DATASETS
# ----------------------------
@st.cache_data
def load_data():
    try:
        dataset_url = "https://github.com/Volandofernando/Material-Literature-data-/raw/main/Dataset.xlsx"
        survey_url = "https://github.com/Volandofernando/REAL-TIME-Dataset/raw/main/IT%20Innovation%20in%20Fabric%20Industry%20%20(Responses).xlsx"

        df_fabrics = pd.read_excel(dataset_url)
        df_survey = pd.read_excel(survey_url)

        return df_fabrics, df_survey
    except Exception as e:
        st.error(f"❌ Failed to load datasets: {e}")
        return None, None


df_fabrics, df_survey = load_data()

if df_fabrics is None:
    st.stop()

# ----------------------------
# USER INPUTS
# ----------------------------
st.title("👕 SweatSmart AI – Fabric Comfort Recommender")
st.markdown(
    "### AI-Powered Fabric Comfort Recommender\n"
    "Trusted by textile R&D, apparel design, and sportswear innovation teams. "
    "Adjust your conditions and instantly see top fabric recommendations optimized "
    "for **comfort, sweat management, and performance.**"
)

col1, col2, col3 = st.columns(3)
with col1:
    temp = st.slider("🌡️ Temperature (°C)", 10, 45, 25)
with col2:
    humidity = st.slider("💧 Humidity (%)", 20, 100, 60)
with col3:
    activity = st.selectbox("🏃 Activity Level", ["Low", "Medium", "High"])

activity_map = {"Low": 1, "Medium": 2, "High": 3}

# ----------------------------
# AI RECOMMENDER
# ----------------------------
def recommend_fabrics(temp, humidity, activity):
    try:
        features = df_fabrics[["Temperature", "Humidity", "Activity"]]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        model = NearestNeighbors(n_neighbors=3)
        model.fit(scaled_features)

        query = scaler.transform([[temp, humidity, activity_map[activity]]])
        distances, indices = model.kneighbors(query)

        return df_fabrics.iloc[indices[0]]
    except Exception as e:
        st.error(f"⚠️ Recommendation error: {e}")
        return pd.DataFrame()


recommendations = recommend_fabrics(temp, humidity, activity)

if not recommendations.empty:
    st.subheader("✅ Top Fabric Recommendations")
    for _, row in recommendations.iterrows():
        fabric = row["Fabric"]
        st.markdown(f"**{fabric}** – {FABRIC_INFO.get(fabric, 'No description available')}")

    # ----------------------------
    # EXPORT OPTIONS
    # ----------------------------
    st.subheader("📤 Export Recommendations")

    # Excel
    excel_buffer = io.BytesIO()
    recommendations.to_excel(excel_buffer, index=False, engine="xlsxwriter")
    st.download_button(
        "⬇️ Download as Excel",
        data=excel_buffer,
        file_name="fabric_recommendations.xlsx",
        mime="application/vnd.ms-excel"
    )

    # PDF
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "Fabric Comfort Recommendations")
    c.setFont("Helvetica", 12)

    y = 760
    for _, row in recommendations.iterrows():
        fabric = row["Fabric"]
        desc = FABRIC_INFO.get(fabric, "No description available")
        c.drawString(80, y, f"{fabric}: {desc}")
        y -= 20

    c.save()
    st.download_button(
        "⬇️ Download as PDF",
        data=pdf_buffer.getvalue(),
        file_name="fabric_recommendations.pdf",
        mime="application/pdf"
    )
