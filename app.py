# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# --- Utils (assumed present in your project) ---
from utils import load_config, load_datasets, detect_features_and_target, train_model, evaluate_model

# -------------------------------
# Load config early and set page config before other Streamlit calls
# -------------------------------
config = load_config()
st.set_page_config(page_title=config["app"]["title"], layout="wide", page_icon="👕")

# Optional site verification meta
st.markdown(
    """<meta name="google-site-verification" content="_NhjPZ3SK1IoAqj4b04D7AlhSSPzpgfZSjmuZq3nE9E" />""",
    unsafe_allow_html=True,
)

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🤖 SweatyBot"])

if page == "🏠 Home":
    st.write("Welcome to SweatSmart AI Fabrics!")

if page == "🤖 SweatyBot":
    # try to import/render sweatybot if available; otherwise show a message
    try:
        import sweatybot
        sweatybot.render()
    except Exception:
        st.info("SweatyBot module not found or raised an error. Continue with core app.")

# -------------------------------
# Styling
# -------------------------------
st.markdown(
    f"""
<style>
    .main {{
        background-color: #0E1117;
        color: #EAEAEA;
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
""",
    unsafe_allow_html=True,
)

# App title + subtitle
st.title(f"👕 {config['app']['title']}")
st.subheader("Next-Gen Comfort & Performance Recommender for the Apparel Industry")

# -------------------------------
# Fabric knowledge base (static)
# -------------------------------
fabric_info = {
    "Cotton": "Breathable, soft, and moisture-absorbent. Ideal for summer and casual wear.",
    "Polyester": "Durable, lightweight, quick-drying, but less breathable. Common in sportswear.",
    "Nylon": "Strong, elastic, and abrasion-resistant. Often used in activewear and outerwear.",
    "Wool": "Warm, insulating, and moisture-wicking. Perfect for cold climates.",
    "Silk": "Luxurious, smooth, and breathable. Popular for formal or premium garments.",
    "Linen": "Highly breathable, lightweight, and cooling. Excellent for hot weather.",
    "Rayon": "Soft and versatile with a silk-like feel. Used in both fashion and performance fabrics.",
    "Spandex": "Stretchable and elastic. Blended with other fibers for comfort and flexibility.",
}

# -------------------------------
# Intro copy
# -------------------------------
st.markdown(
    """
<div class="intro-box">
    <h3>AI-Powered Fabric Recommender</h3>
    <p>
    Designed for <b>textile manufacturers</b>, <b>sportswear innovators</b>, and <b>fashion R&D labs</b>.  
    Powered by <b>machine learning</b> trained on fabric performance data and thermophysiological models.
    </p>
    <p>
    Enter your environmental conditions and get <b>optimized fabric recommendations</b> 
    balancing <b>comfort, sweat control, and performance</b>.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------
# Load Data & Train Model (robust handling)
# -------------------------------
try:
    df = load_datasets(config)
except Exception as e:
    st.error(f"❌ Failed to load datasets: {e}")
    st.stop()

feature_cols, target_col = detect_features_and_target(df, config)

if target_col is None or len(feature_cols) < 1:
    st.error("❌ Dataset error: required features/target not found!")
    st.stop()

try:
    model, scaler, X_test, y_test, df_clean = train_model(df, feature_cols, target_col, config)
except Exception as e:
    st.error(f"❌ Model training failed: {e}")
    st.stop()

# Make a safe copy to avoid SettingWithCopy warnings
df_clean = df_clean.copy()

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📌 AI Comfort Recommender", "📊 Insights", "🤖 Model Performance", "ℹ️ About"]
)

# -------------------------------
# TAB 1: Recommendation
# -------------------------------
with tab1:
    with st.sidebar.expander("⚙️ Set Environment Conditions", expanded=True):
        st.markdown("Adjust the parameters to simulate **real-world wearing scenarios**:")

        temperature = st.slider(
            "🌡️ Outdoor Temperature (°C)", 10, 45, 28, help="Higher temperatures increase thermal stress."
        )
        humidity = st.slider(
            "💧 Humidity (%)", 10, 100, 60, help="High humidity slows sweat evaporation → feels warmer."
        )
        sweat_sensitivity = st.select_slider(
            "🧍 Sweat Sensitivity", ["Low", "Medium", "High"], value="Medium", help="How easily the wearer sweats."
        )
        activity_intensity = st.select_slider(
            "🏃 Activity Intensity", ["Low", "Moderate", "High"], value="Moderate", help="Higher activity = more sweat."
        )

    sweat_map = {"Low": 1, "Medium": 2, "High": 3}
    activity_map = {"Low": 1, "Moderate": 2, "High": 3}
    sweat_num, activity_num = sweat_map[sweat_sensitivity], activity_map[activity_intensity]

    # Build user input vector — adapt if your model expects different feature order
    user_input = np.array(
        [
            [
                sweat_num * 5,
                800 + humidity * 5,
                60 + activity_num * 10,
                0.04 + (temperature - 25) * 0.001,
            ]
        ]
    )

    # Scale safely (ensure scaler exists and supports transform)
    try:
        user_input_scaled = scaler.transform(user_input)
    except Exception as e:
        st.error(f"❌ Failed to scale input: {e}")
        st.stop()

    # Predict comfort score
    try:
        predicted_score = float(model.predict(user_input_scaled)[0])
    except Exception as e:
        st.error(f"❌ Model prediction failed: {e}")
        st.stop()

    # Normalize to 0-100
    min_score = float(df_clean[target_col].min())
    max_score = float(df_clean[target_col].max())
    if max_score == min_score:
        predicted_percent = 0.0
    else:
        predicted_percent = round(((predicted_score - min_score) / (max_score - min_score)) * 100, 1)
        predicted_percent = max(0, min(predicted_percent, 100))

    # Industrial weighting adjustments (work on a copy)
    df_clean["comfort_weighted"] = df_clean[target_col].astype(float)

    if humidity > 70:
        df_clean["comfort_weighted"] += 0.05 * humidity
    if temperature > 32:
        df_clean["comfort_weighted"] += 0.03 * temperature
    if sweat_sensitivity == "High":
        df_clean["comfort_weighted"] += 5
    if activity_intensity == "High":
        df_clean["comfort_weighted"] += 2

    df_clean["predicted_diff"] = (df_clean["comfort_weighted"] - predicted_score).abs()

    top_matches = df_clean.sort_values(by=["predicted_diff", "comfort_weighted"], ascending=[True, False]).head(3)

    def generate_fabric_explanation(fabric, temperature, humidity, sweat_sensitivity, activity_intensity):
        base = f"{fabric} is recommended based on its adaptive performance under current climate and activity levels."
        if "Cotton" in fabric:
            base += " It provides high breathability and moisture absorption, keeping the body cool."
        elif "Polyester" in fabric:
            base += " It offers durability and quick-dry properties, suitable for sportswear."
        elif "Nylon" in fabric:
            base += " It maintains elasticity and strength under pressure, ideal for high-activity use."
        elif "Wool" in fabric:
            base += " It insulates while allowing vapor transmission, good for cooler climates."
        elif "Linen" in fabric:
            base += " It ensures rapid heat dissipation and comfort in hot, humid environments."
        elif "Silk" in fabric:
            base += " It provides thermal regulation and a soft luxurious texture."
        elif "Rayon" in fabric:
            base += " It mimics natural fibers while offering moisture absorption and good drape."
        elif "Spandex" in fabric:
            base += " It introduces flexibility and stretch, enhancing comfort during movement."
        else:
            base += " It demonstrates balanced heat and moisture management characteristics."

        if temperature > 32 and humidity > 70:
            base += " Its evaporative cooling and moisture-wicking ability make it suitable for tropical conditions."
        elif temperature < 20:
            base += " The fabric’s thermal retention properties enhance comfort in cooler climates."
        if sweat_sensitivity == "High":
            base += " Its air-permeable structure reduces discomfort from perspiration."
        if activity_intensity == "High":
            base += " The material allows rapid moisture evaporation, supporting performance efficiency."

        return base

    # Display results
    st.metric("Predicted Comfort Index", f"{predicted_percent} %", help="Normalized comfort score across 0–100 scale")

    st.markdown("## 🔹 Recommended Fabrics for Your Scenario")

    # Prepare recommendations list (safe fallback for fabric_type)
    recommendations = []
    for _, row in top_matches.iterrows():
        fabric = row.get("fabric_type", "Unknown Fabric")
        score_raw = float(row[target_col])
        if max_score == min_score:
            normalized_score = 0.0
        else:
            normalized_score = ((score_raw - min_score) / (max_score - min_score)) * 100
            normalized_score = float(np.clip(normalized_score, 0, 100))
        comfort_label = f"{normalized_score:.1f} %"
        explanation = generate_fabric_explanation(fabric, temperature, humidity, sweat_sensitivity, activity_intensity)
        recommendations.append({"Fabric": fabric, "Comfort Score (%)": normalized_score, "Explanation": explanation})

    # Display recommendation cards in up to 3 columns
    num_cols = min(3, max(1, len(recommendations)))
    cols = st.columns(num_cols)
    for i, rec in enumerate(recommendations):
        with cols[i % num_cols]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>🧵 {rec['Fabric']}</h4>
                    <div class="metric-value">{rec['Comfort Score (%)']:.1f} %</div>
                    <div class="metric-label">Comfort Score</div>
                    <p>{rec['Explanation']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Chart (Altair) — be explicit about color with alt.value()
    chart_data = pd.DataFrame(recommendations)
    if not chart_data.empty:
        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(x=alt.X("Fabric:N", sort=None), y=alt.Y("Comfort Score (%)", title="Comfort (%)"))
            .properties(width="container")
        )
        # set a single-color encoding via mark properties
        chart = chart.configure_mark(color=config["app"]["theme_color"])
        st.altair_chart(chart, use_container_width=True)

    # -------------------------------
    # Export Functions
    # -------------------------------
    st.markdown("### 📤 Export Recommendation Report")

    # Excel
    excel_buffer = BytesIO()
    pd.DataFrame(recommendations).to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    st.download_button(
        label="📊 Download Excel Report",
        data=excel_buffer.getvalue(),
        file_name="fabric_recommendations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # PDF
    def generate_pdf(recs):
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Fabric Recommendation Report")
        c.setFont("Helvetica", 12)
        y = height - 100
        for rec in recs:
            c.drawString(50, y, f"Fabric: {rec['Fabric']}  |  Comfort Score: {rec['Comfort Score (%)']:.1f}%")
            y -= 18
            c.setFont("Helvetica-Oblique", 10)
            # wrap long explanation manually if needed
            lines = str(rec["Explanation"]).split(". ")
            for line in lines:
                c.drawString(70, y, line.strip()[:120])
                y -= 12
            c.setFont("Helvetica", 12)
            y -= 12
            if y < 80:
                c.showPage()
                y = height - 80
        c.save()
        pdf_buffer.seek(0)
        return pdf_buffer

    pdf_report = generate_pdf(recommendations)
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_report.getvalue(),
        file_name="fabric_recommendations.pdf",
        mime="application/pdf",
    )

    # Extra info table
    st.markdown("### 🧵 Fabric Knowledge Base")
    st.dataframe(pd.DataFrame(fabric_info.items(), columns=["Fabric", "Description"]))

# -------------------------------
# TAB 2: Dataset Insights
# -------------------------------
with tab2:
    st.markdown("### 📊 Dataset Insights")
    st.markdown("#### 🔍 Preview of Fabric Dataset")
    st.dataframe(df_clean.head(10))

    st.markdown("#### 📈 Summary Statistics")
    st.write(df_clean.describe(include="all"))

    # Correlation heatmap (features vs comfort)
    try:
        corr = df_clean[feature_cols + [target_col]].corr().reset_index().melt("index")
        heatmap = (
            alt.Chart(corr)
            .mark_rect()
            .encode(
                x=alt.X("index:O"),
                y=alt.Y("variable:O"),
                color=alt.Color("value:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["index", "variable", alt.Tooltip("value:Q", format=".2f")],
            )
        )
        st.altair_chart(heatmap, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render correlation heatmap: {e}")

    # Top comfort-performing fabrics
    if "fabric_type" in df_clean.columns:
        st.markdown("#### 🧵 Top Comfort-Performing Fabrics")
        top_fabrics = df_clean.groupby("fabric_type")[target_col].mean().reset_index()
        top_fabrics = top_fabrics.sort_values(by=target_col, ascending=False).head(5)
        bar_chart = (
            alt.Chart(top_fabrics)
            .mark_bar()
            .encode(
                x=alt.X("fabric_type:N", sort=None, title="Fabric"),
                y=alt.Y(f"{target_col}:Q", title="Average Comfort Score"),
                tooltip=["fabric_type", alt.Tooltip(target_col, format=".2f")],
            )
        )
        st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.info("Dataset does not include 'fabric_type' column to show top fabrics.")

# -------------------------------
# TAB 3: Model Performance
# -------------------------------
with tab3:
    try:
        metrics = evaluate_model(model, X_test, y_test)
        r2 = metrics.get("r2", float("nan"))
        rmse = metrics.get("rmse", float("nan"))
        st.metric("R² Score", f"{r2:.2f}" if pd.notna(r2) else "N/A")
        with st.expander("ℹ️ What is R² Score?"):
            if pd.notna(r2):
                st.write(f"R² = {r2:.2f} → The model explains about {r2*100:.1f}% of the comfort variation.")
            else:
                st.write("R² not available.")
        st.metric("RMSE", f"{rmse:.2f}" if pd.notna(rmse) else "N/A")
    except Exception as e:
        st.error(f"Could not evaluate model performance: {e}")

# -------------------------------
# TAB 4: About
# -------------------------------
with tab4:
    st.markdown(
        f"""
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
    """
    )

st.markdown(
    """
---
🔗 **Project Repository:** [GitHub – VolandoFernando/sweatsmart-ai](https://github.com/VolandoFernando/sweatsmart-ai)  
📘 **Author:** Volando Fernando | University of West London (UWL)
"""
)
