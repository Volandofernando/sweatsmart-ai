import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from utils import load_config, load_datasets, detect_features_and_target, train_model, evaluate_model
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# -------------------------------
# Load Config
# -------------------------------
config = load_config()
st.set_page_config(page_title=config["app"]["title"], layout="wide")

# -------------------------------
# Fabric Knowledge Base
# -------------------------------
fabric_info = {
    "Cotton": "Breathable, soft, and moisture-absorbent. Common for casual and summer wear.",
    "Polyester": "Durable, lightweight, quick-drying, but less breathable. Often used in sportswear.",
    "Nylon": "Strong, elastic, and abrasion-resistant. Common in activewear and outerwear.",
    "Wool": "Warm, insulating, and moisture-wicking. Suitable for cold climates.",
    "Silk": "Luxurious, smooth, and breathable. Often used for formal or premium garments.",
    "Linen": "Highly breathable, lightweight, and cooling. Ideal for hot weather.",
    "Rayon": "Soft and versatile with silk-like feel. Used in both fashion and performance fabrics.",
    "Spandex": "Stretchable and elastic. Often blended with other fibers for comfort and flexibility."
}

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown(f"""
<style>
    .main {{ background-color: #FAFAFA; }}
    h1, h2, h3 {{ color: {config['app']['theme_color']}; font-family: 'Helvetica Neue', sans-serif; }}
    .intro-box {{
        padding: 1.2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
        box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    }}
    .metric-card {{
        background: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 1rem;
    }}
    .metric-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #1F2937;
    }}
    .metric-label {{
        font-size: 0.9rem;
        color: #6B7280;
    }}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title + Branding
# -------------------------------
st.title(f"👕 {config['app']['title']}")
st.subheader("Next-Gen Comfort & Performance Recommender for the Apparel Industry")

# -------------------------------
# Intro Section
# -------------------------------
st.markdown("""
<div class="intro-box">
    <h3>AI-Powered Fabric Recommender</h3>
    <p>
    Designed for <b>textile manufacturers</b>, <b>sportswear innovators</b>, and <b>fashion R&D labs</b>.  
    Powered by <b>machine learning</b> trained on fabric performance data and thermophysiological models.
    </p>
    <p>
    Enter your environmental conditions and instantly receive <b>optimized fabric recommendations</b> 
    with detailed explanations, balancing <b>comfort, sweat control, and performance</b>.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Load Data & Train Model
# -------------------------------
try:
    df = load_datasets(config)
except Exception as e:
    st.error(f"❌ Failed to load datasets: {e}")
    st.stop()

feature_cols, target_col = detect_features_and_target(df, config)

if target_col is None or len(feature_cols) < 4:
    st.error("❌ Dataset error: required features/target not found!")
    st.stop()

model, scaler, X_test, y_test, df_clean = train_model(df, feature_cols, target_col, config)

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Recommender", "📊 Insights", "🤖 Model Performance", "ℹ️ About"])

# -------------------------------
# TAB 1: Recommendation
# -------------------------------
with tab1:
    with st.sidebar.expander("⚙️ Set Environment Conditions", expanded=True):
        st.markdown("Adjust the parameters to simulate **real-world wearing scenarios**.")
        temperature = st.slider("🌡️ Outdoor Temperature (°C)", 10, 45, 28)
        humidity = st.slider("💧 Humidity (%)", 10, 100, 60)
        sweat_sensitivity = st.select_slider("🧍 Sweat Sensitivity", ["Low", "Medium", "High"])
        activity_intensity = st.select_slider("🏃 Activity Intensity", ["Low", "Moderate", "High"])

    sweat_map = {"Low": 1, "Medium": 2, "High": 3}
    activity_map = {"Low": 1, "Moderate": 2, "High": 3}
    sweat_num, activity_num = sweat_map[sweat_sensitivity], activity_map[activity_intensity]

    user_input = np.array([[sweat_num * 5,
                            800 + humidity * 5,
                            60 + activity_num * 10,
                            0.04 + (temperature - 25) * 0.001]])
    user_input_scaled = scaler.transform(user_input)

    predicted_score = model.predict(user_input_scaled)[0]
    df_clean["predicted_diff"] = abs(df_clean[target_col] - predicted_score)
    top_matches = df_clean.sort_values(by="predicted_diff").head(3)

    st.markdown("## 🔹 Recommended Fabrics for Your Scenario")
    cols = st.columns(3)
    recommendations = []

    for i, (_, row) in enumerate(top_matches.iterrows()):
        fabric = row.get("fabric_type", "Unknown")
        explanation = fabric_info.get(fabric, "No description available.")
        score = round(row[target_col], 2)

        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🧵 {fabric}</h4>
                <div class="metric-value">{score}</div>
                <div class="metric-label">Comfort Score</div>
                <p>{explanation}</p>
            </div>
            """, unsafe_allow_html=True)

        recommendations.append({
            "Fabric": fabric,
            "Comfort Score": score,
            "Explanation": explanation
        })

    # Chart
    chart_data = pd.DataFrame(recommendations)
    chart = alt.Chart(chart_data).mark_bar(color=config["app"]["theme_color"]).encode(
        x=alt.X("Fabric", sort=None),
        y="Comfort Score"
    )
    st.altair_chart(chart, use_container_width=True)

    # -------------------------------
    # Export Functions
    # -------------------------------
    st.markdown("### 📤 Export Recommendation Report")

    # Export to Excel
    excel_buffer = BytesIO()
    pd.DataFrame(recommendations).to_excel(excel_buffer, index=False)
    st.download_button(
        label="📊 Download Excel Report",
        data=excel_buffer.getvalue(),
        file_name="fabric_recommendations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Export to PDF
    def generate_pdf(recommendations):
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        width, height = A4

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Fabric Recommendation Report")

        c.setFont("Helvetica", 12)
        y = height - 100
        for rec in recommendations:
            c.drawString(50, y, f"Fabric: {rec['Fabric']}  |  Comfort Score: {rec['Comfort Score']}")
            y -= 20
            c.setFont("Helvetica-Oblique", 10)
            c.drawString(70, y, f"Details: {rec['Explanation']}")
            c.setFont("Helvetica", 12)
            y -= 30

        c.save()
        pdf_buffer.seek(0)
        return pdf_buffer

    pdf_report = generate_pdf(recommendations)
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_report,
        file_name="fabric_recommendations.pdf",
        mime="application/pdf"
    )

# -------------------------------
# TAB 2: Dataset Insights
# -------------------------------
with tab2:
    st.markdown("### 📊 Dataset Overview")
    st.dataframe(df_clean.head(10))

    st.write("#### Summary Statistics")
    st.write(df_clean.describe())

    st.write("#### Correlation Heatmap")
    corr = df_clean[feature_cols + [target_col]].corr().reset_index().melt("index")
    heatmap = alt.Chart(corr).mark_rect().encode(
        x="index:O", y="variable:O", color="value:Q"
    )
    st.altair_chart(heatmap, use_container_width=True)

# -------------------------------
# TAB 3: Model Performance
# -------------------------------
with tab3:
    metrics = evaluate_model(model, X_test, y_test)
    st.metric("R² Score", metrics["r2"])
    st.metric("RMSE", metrics["rmse"])

    st.write("#### Feature Importances")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    feat_chart = alt.Chart(feat_df).mark_bar(color=config["app"]["theme_color"]).encode(
        x="Feature",
        y="Importance"
    )
    st.altair_chart(feat_chart, use_container_width=True)

# -------------------------------
# TAB 4: About
# -------------------------------
with tab4:
    st.markdown(f"""
    **{config['app']['title']}**  
    A professional AI system for **fabric comfort and performance recommendation**.  

    🚀 Key Features:  
    - AI-powered comfort prediction based on fabric & environment  
    - Built-in **knowledge base** explaining each fabric type  
    - Exportable reports in **Excel and PDF**  
    - Optimized for **R&D, apparel design, and sportswear innovation**  

    🌍 Industry Use Cases:  
    - **Sportswear brands**: digital testing of fabrics before production  
    - **Fashion houses**: optimize seasonal fabric selection  
    - **Healthcare textiles**: patient comfort and hospital uniforms  

    👨‍💻 Built by: *Volando Fernando*  
    """)
