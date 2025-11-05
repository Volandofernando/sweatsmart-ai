import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from utils import load_config, load_datasets, detect_features_and_target, train_model, evaluate_model
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.markdown("""
<meta name="google-site-verification" content="_NhjPZ3SK1IoAqj4b04D7AlhSSPzpgfZSjmuZq3nE9E" />
""", unsafe_allow_html=True)

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🤖 SweatyBot"])

if page == "🏠 Home":
    st.write("Welcome to SweatSmart AI Fabrics!")
if page == "🤖 SweatyBot":
    import sweatybot
    sweatybot.render()

# -------------------------------
# Load Config
# -------------------------------
config = load_config()
st.set_page_config(page_title=config["app"]["title"], layout="wide")

# -------------------------------
# Fabric Knowledge Base
# -------------------------------
fabric_info = {
    "Cotton": "Breathable, soft, and moisture-absorbent. Ideal for summer and casual wear.",
    "Polyester": "Durable, lightweight, quick-drying, but less breathable. Common in sportswear.",
    "Nylon": "Strong, elastic, and abrasion-resistant. Often used in activewear and outerwear.",
    "Wool": "Warm, insulating, and moisture-wicking. Perfect for cold climates.",
    "Silk": "Luxurious, smooth, and breathable. Popular for formal or premium garments.",
    "Linen": "Highly breathable, lightweight, and cooling. Excellent for hot weather.",
    "Rayon": "Soft and versatile with a silk-like feel. Used in both fashion and performance fabrics.",
    "Spandex": "Stretchable and elastic. Blended with other fibers for comfort and flexibility."
}

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown(f"""
<style>
    .main {{
        background-color: #0E1117;  /* Dark background */
        color: #EAEAEA;  /* Light text */
        font-family: 'Helvetica Neue', sans-serif;
    }}
    h1, h2, h3 {{
        color: {config['app']['theme_color']};
        font-weight: 700;
    }}
    .intro-box {{
        padding: 1.2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #1E1E1E 0%, #2A2A2A 100%);
        box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
        color: #F5F5F5;
    }}
    .metric-card {{
        background: #1C1C1C;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }}
    .metric-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {config['app']['theme_color']};
    }}
    .metric-label {{
        font-size: 0.9rem;
        color: #A0A0A0;
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
tab1, tab2, tab3, tab4 = st.tabs(["📌 AI Comfort Recommender", "📊 Insights", "🤖 Model Performance", "ℹ️ About"])

# -------------------------------
# TAB 1: Recommendation
# -------------------------------
with tab1:
    with st.sidebar.expander("⚙️ Set Environment Conditions", expanded=True):
        st.markdown("Adjust the parameters to simulate **real-world wearing scenarios**:")

        temperature = st.slider("🌡️ Outdoor Temperature (°C)", 10, 45, 28,
                                help="Higher temperatures increase thermal stress and impact fabric comfort.")
        humidity = st.slider("💧 Humidity (%)", 10, 100, 60,
                             help="Humidity = moisture in the air. High humidity slows sweat evaporation → fabrics feel warmer.")
        sweat_sensitivity = st.select_slider("🧍 Sweat Sensitivity", ["Low", "Medium", "High"],
                                             help="Represents how easily you sweat during activities.")
        activity_intensity = st.select_slider("🏃 Activity Intensity", ["Low", "Moderate", "High"],
                                              help="Higher activity = more heat and sweat.")

    sweat_map = {"Low": 1, "Medium": 2, "High": 3}
    activity_map = {"Low": 1, "Moderate": 2, "High": 3}
    sweat_num, activity_num = sweat_map[sweat_sensitivity], activity_map[activity_intensity]

    user_input = np.array([[sweat_num * 5,
                            800 + humidity * 5,
                            60 + activity_num * 10,
                            0.04 + (temperature - 25) * 0.001]])
    user_input_scaled = scaler.transform(user_input)

# -------------------------------
# 🧠 Enhanced Comfort Prediction & AI Reasoning
# -------------------------------
# Predict comfort score
    predicted_score = float(model.predict(user_input_scaled)[0])
    
    # Normalize prediction to 0–100 range for interpretability
    min_score = float(df_clean[target_col].min())
    max_score = float(df_clean[target_col].max())
    predicted_percent = round(((predicted_score - min_score) / (max_score - min_score)) * 100, 1)
    predicted_percent = max(0, min(predicted_percent, 100))  # clamp 0–100
    
    # --- Industrial weighting adjustments ---
    df_clean["comfort_weighted"] = df_clean[target_col]
    
    if humidity > 70:
        df_clean["comfort_weighted"] += 0.05 * humidity
    if temperature > 32:
        df_clean["comfort_weighted"] += 0.03 * temperature
    if sweat_sensitivity == "High":
        df_clean["comfort_weighted"] += 5
    if activity_intensity == "High":
        df_clean["comfort_weighted"] += 2
    
    # --- Ranking fabrics ---
    df_clean["predicted_diff"] = abs(df_clean["comfort_weighted"] - predicted_score)
    top_matches = df_clean.sort_values(by=["predicted_diff", "comfort_weighted"], ascending=[True, False]).head(3)
    
    # --- AI-driven explanation generator ---
    # --- Human-Friendly Comfort Explanation ---
    def generate_fabric_explanation(fabric, score_percent):
    """
    Converts numerical comfort score into a simple real-life meaning
    that any normal clothing buyer can understand.
    """
    if score_percent >= 75:
        comfort_description = "very comfortable and ideal for the current weather"
    elif score_percent >= 50:
        comfort_description = "reasonably comfortable for general daily wear"
    else:
        comfort_description = "may feel warm or less breathable in this weather"

    return (
        f"{fabric} is rated as **{comfort_description}**. "
        "This score reflects heat control, sweat evaporation, and how breathable the fabric feels on skin."
    )

    
        # Adaptive reasoning
        if temperature > 32 and humidity > 70:
            base += " Its evaporative cooling and moisture-wicking ability make it suitable for tropical conditions."
        elif temperature < 20:
            base += " The fabric’s thermal retention properties enhance comfort in cooler climates."
        if sweat_sensitivity == "High":
            base += " Its air-permeable structure reduces discomfort from perspiration."
        if activity_intensity == "High":
            base += " The material allows rapid moisture evaporation, supporting performance efficiency."
    
        return base
    
    # --- Display AI summary ---
    st.metric("Predicted Comfort Index", f"{predicted_percent} %", help="Normalized comfort score across 0–100 scale")
    
    # --- Display top 3 fabric recommendations ---
    st.markdown("## 🔹 Recommended Fabrics for Your Scenario")
    cols = st.columns(3)
    recommendations = []
    
    for i, (_, row) in enumerate(top_matches.iterrows()):
        fabric = row.get("fabric_type", "Unknown")
        score_raw = row[target_col]
        comfort_label = f"{round(score_raw * 100, 1)} %"
        explanation = generate_fabric_explanation(fabric, temperature, humidity, sweat_sensitivity, activity_intensity)
    
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🧵 {fabric}</h4>
                <div class="metric-value">{comfort_label}</div>
                <div class="metric-label">Comfort Score</div>
                <p>{explanation}</p>
            </div>
            """, unsafe_allow_html=True)
    
        recommendations.append({
            "Fabric": fabric,
            "Comfort Score (%)": comfort_label,
            "Explanation": explanation
        })

# Chart
    chart_data = pd.DataFrame(recommendations)
    chart_data["Comfort Score (%)"] = chart_data["Comfort Score (%)"].str.replace("%", "").astype(float)
    chart = alt.Chart(chart_data).mark_bar(color=config["app"]["theme_color"]).encode(
        x=alt.X("Fabric", sort=None),
        y=alt.Y("Comfort Score (%)", title="Comfort (%)")
    )
    st.altair_chart(chart, use_container_width=True)

    # -------------------------------
    # Export Functions
    # -------------------------------
    st.markdown("### 📤 Export Recommendation Report")

    # Excel
    excel_buffer = BytesIO()
    pd.DataFrame(recommendations).to_excel(excel_buffer, index=False)
    st.download_button(
        label="📊 Download Excel Report",
        data=excel_buffer.getvalue(),
        file_name="fabric_recommendations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # PDF
    def generate_pdf(recommendations):
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        width, height = A4

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Fabric Recommendation Report")

        c.setFont("Helvetica", 12)
        y = height - 100
        for rec in recommendations:
            c.drawString(50, y, f"Fabric: {rec['Fabric']}  |  Comfort Score: {rec['Comfort Score (%)']}")
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
    # Extra Info Table
    # -------------------------------
    st.markdown("### 🧵 Fabric Knowledge Base")
    st.dataframe(pd.DataFrame(fabric_info.items(), columns=["Fabric", "Description"]))

# -------------------------------
# TAB 2: Dataset Insights
# -------------------------------
with tab2:
    st.markdown("### 📊 Dataset Insights")

    # Dataset Preview
    st.markdown("#### 🔍 Preview of Fabric Dataset")
    st.dataframe(df_clean.head(10))

    # Summary Statistics
    st.markdown("#### 📈 Summary Statistics")
    st.write(df_clean.describe())

    # Correlation Heatmap
    st.markdown("#### 🔥 Correlation Heatmap (Features vs Comfort)")
    corr = df_clean[feature_cols + [target_col]].corr().reset_index().melt("index")
    heatmap = alt.Chart(corr).mark_rect().encode(
        x="index:O",
        y="variable:O",
        color=alt.Color("value:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["index", "variable", alt.Tooltip("value:Q", format=".2f")]
    )
    st.altair_chart(heatmap, use_container_width=True)

    # Top Comfort-Driving Fabrics
    st.markdown("#### 🧵 Top Comfort-Performing Fabrics")
    top_fabrics = df_clean.groupby("fabric_type")[target_col].mean().reset_index()
    top_fabrics = top_fabrics.sort_values(by=target_col, ascending=False).head(5)
    bar_chart = alt.Chart(top_fabrics).mark_bar(color=config["app"]["theme_color"]).encode(
        x=alt.X("fabric_type", sort=None, title="Fabric"),
        y=alt.Y(target_col, title="Average Comfort Score"),
        tooltip=["fabric_type", alt.Tooltip(target_col, format=".2f")]
    )
    st.altair_chart(bar_chart, use_container_width=True)


# -------------------------------
# TAB 3: Model Performance
# -------------------------------
with tab3:
    metrics = evaluate_model(model, X_test, y_test)
    r2 = metrics["r2"]
    rmse = metrics["rmse"]

    st.metric("R² Score", f"{r2:.2f}")
    with st.expander("ℹ️ What is R² Score?"):
        st.write(f"R² = {r2:.2f} → The model explains about {r2*100:.1f}% of the comfort variation.")
        st.write("Beginner tip: closer to 1 = better, closer to 0 = weak.")

    st.metric("RMSE", f"{rmse:.2f}")
    with st.expander("ℹ️ What is RMSE?"):
        st.write(f"RMSE = {rmse:.2f} → On average, predictions are off by ±{rmse:.2f} units of comfort score.")
        st.write("Beginner tip: lower is better.")

# -------------------------------
# TAB 4: About
# -------------------------------
with tab4:
    st.markdown(f"""
    ## ℹ️ About {config['app']['title']}
    
    ### 🎯 Purpose  
    A professional AI system for **fabric comfort and performance recommendation**.  

    ### 🧵 Supported Fabrics  
    Cotton, Polyester, Nylon, Wool, Silk, Linen, Rayon, Spandex  

    ### 📊 How to Use  
    1. Adjust environment conditions (temperature, humidity, activity, sweat sensitivity).  
    2. View top recommended fabrics with **comfort percentages**.  
    3. Download professional reports in **Excel or PDF**.  
    4. Explore the built-in **Fabric Knowledge Base**.  

    ### 🚀 Industry Applications  
    - **Sportswear brands** → test fabrics digitally before production  
    - **Fashion houses** → seasonal fabric optimization  
    - **Healthcare textiles** → patient comfort and uniforms  

    👨💻 Built by: *Volando Fernando*
    """)

st.markdown("""
---
🔗 **Project Repository:** [GitHub – VolandoFernando/sweatsmart-ai](https://github.com/VolandoFernando/sweatsmart-ai)  
📘 **Author:** Volando Fernando | University of West London (UWL)
""")
