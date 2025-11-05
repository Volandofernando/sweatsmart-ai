# app.py (replace your existing app.py with this file)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from utils import load_config, load_datasets, detect_features_and_target, train_model, evaluate_model
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import math

# -------------------------------
# Page setup
# -------------------------------
config = load_config()
st.set_page_config(page_title=config["app"]["title"], layout="wide")

st.markdown("""
<meta name="google-site-verification" content="_NhjPZ3SK1IoAqj4b04D7AlhSSPzpgfZSjmuZq3nE9E" />
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar / Navigation
# -------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🤖 SweatyBot"])

if page == "🏠 Home":
    st.write("Welcome to SweatSmart AI Fabrics!")
if page == "🤖 SweatyBot":
    import sweatybot
    sweatybot.render()

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
# Styling
# -------------------------------
st.markdown(f"""
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
        padding: 1.2rem; border-radius: 12px;
        background: linear-gradient(135deg,#1E1E1E 0%,#2A2A2A 100%);
        box-shadow: 0px 4px 12px rgba(0,0,0,0.25); color: #F5F5F5;
    }}
    .metric-card {{ background: #1C1C1C; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.3); text-align:center; color:#FFF; }}
    .metric-value {{ font-size:1.5rem; font-weight:700; color:{config['app']['theme_color']}; }}
    .metric-label {{ font-size:0.9rem; color:#A0A0A0; }}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.title(f"👕 {config['app']['title']}")
st.subheader("Next-Gen Comfort & Performance Recommender for the Apparel Industry")
st.markdown("""
<div class="intro-box">
  <h3>AI-Powered Fabric Recommender</h3>
  <p>Designed for textile manufacturers, sportswear innovators and fashion R&D. Enter conditions and receive recommended fabrics with 0–100 comfort (%) and human-friendly explanations.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Load datasets + model
# -------------------------------
try:
    df = load_datasets(config)
except Exception as e:
    st.error(f"❌ Failed to load datasets: {e}")
    st.stop()

feature_cols, target_col = detect_features_and_target(df, config)
if target_col is None or len(feature_cols) < 1:
    st.error("❌ Dataset error: target/features not found. Check config and dataset.")
    st.stop()

# train_model returns (model, scaler, X_test, y_test, df)
model, scaler, X_test, y_test, df_clean = train_model(df, feature_cols, target_col, config)

# defensive copy (do not mutate original)
df_clean = df_clean.copy()

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 AI Comfort Recommender", "📊 Insights", "🤖 Model Performance", "ℹ️ About"])

# -------------------------------
# Helper: Normalize numeric array to [0,1] safely
# -------------------------------
def minmax_scale_value(x, xmin, xmax):
    if xmax == xmin:
        return 0.0
    return (x - xmin) / (xmax - xmin)

# -------------------------------
# Human-friendly explanation generator
# -------------------------------
def generate_fabric_explanation_from_percent(fabric, percent):
    """
    Simple user-friendly explanation based on percent comfort (0-100).
    Keeps language non-technical for normal users / buyers.
    """
    percent = float(percent)
    if percent >= 80:
        band = "Excellent"
        text = "Highly comfortable — ideal choice for this weather and activity."
    elif percent >= 60:
        band = "Good"
        text = "Comfortable for most users. A reliable choice."
    elif percent >= 40:
        band = "Moderate"
        text = "May be acceptable; expect some warmth or slower drying."
    else:
        band = "Low"
        text = "May feel warm or clingy; not recommended for high-heat or high-sweat activities."

    short = fabric_info.get(fabric, "")
    return f"{band} ({percent:.1f}%). {text} {short}"

# -------------------------------
# TAB 1: Recommendation
# -------------------------------
with tab1:
    with st.sidebar.expander("⚙️ Set Environment Conditions", expanded=True):
        st.markdown("Adjust parameters to simulate real-world wearing scenarios:")
        temperature = st.slider("🌡️ Outdoor Temperature (°C)", 10, 45, 28)
        humidity = st.slider("💧 Humidity (%)", 10, 100, 60)
        sweat_sensitivity = st.select_slider("🧍 Sweat Sensitivity", ["Low", "Medium", "High"])
        activity_intensity = st.select_slider("🏃 Activity Intensity", ["Low", "Moderate", "High"])

    # map sliders to small numeric features (kept safe & documented)
    sweat_map = {"Low": 1, "Medium": 2, "High": 3}
    activity_map = {"Low": 1, "Moderate": 2, "High": 3}
    sweat_num = sweat_map[sweat_sensitivity]
    activity_num = activity_map[activity_intensity]

    # Build input vector of same length as feature_cols:
    # common project assumption: first four engineered features correspond to:
    # [sweat_sensitivity_feature, air_or_perm_like_feature, activity_feature, thermal_feature]
    # If dataset features differ, this still creates a safe vector zero-filled beyond first 4.
    input_vec = np.zeros((1, len(feature_cols)), dtype=float)
    # Populate first 4 slots with engineered numeric values (safe defaults)
    vals = [
        float(sweat_num * 5),                 # sweat sensitivity scaled
        float(800 + humidity * 5),            # synthetic 'air' or related numeric
        float(60 + activity_num * 10),        # synthetic activity measure
        float(0.04 + (temperature - 25) * 0.001)  # small thermal offset
    ]
    for i in range(min(4, len(feature_cols))):
        input_vec[0, i] = vals[i]

    # Scale using the trained scaler
    try:
        user_input_scaled = scaler.transform(input_vec)
    except Exception as e:
        st.error(f"Scaler transform failed: {e}")
        st.stop()

    # Predict
    try:
        predicted_score = float(model.predict(user_input_scaled)[0])
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    # Normalize predicted_score to 0..1 using dataset target bounds
    data_min = float(df_clean[target_col].min())
    data_max = float(df_clean[target_col].max())
    if math.isclose(data_max, data_min):
        normalized_pred = 0.0
    else:
        normalized_pred = (predicted_score - data_min) / (data_max - data_min)

    # Convert to percent 0..100
    predicted_percent = float(np.clip(normalized_pred * 100.0, 0.0, 100.0))

    # -------------------------------
    # Create per-fabric score (0..100) for display and ranking
    # Strategy:
    #  - take original target (assumed numeric), min-max scale to 0..1
    #  - apply small industry adjustments (normalized factor from humidity/temp/activity)
    #  - compute absolute diff from normalized_pred to rank closest-fit fabrics
    # -------------------------------
    df_local = df_clean.copy()
    # Ensure numeric
    df_local[target_col] = pd.to_numeric(df_local[target_col], errors="coerce").fillna(data_min)

    # min-max scale fabric base score to [0,1]
    df_local["_base_norm"] = df_local[target_col].apply(lambda x: minmax_scale_value(x, data_min, data_max))

    # compute small adjustment factors (range roughly -0.1 .. +0.1)
    hum_factor = (humidity - 50) / 500.0        # humidity 50 --> 0.0 ; humidity 100 --> +0.1
    temp_factor = (temperature - 25) / 500.0    # temp 25 --> 0.0 ; temp 45 --> +0.04
    sweat_factor = (sweat_num - 2) / 50.0       # small influence
    activity_factor = (activity_num - 2) / 50.0

    # combine safely (clamp later)
    df_local["_weighted"] = df_local["_base_norm"] + hum_factor + temp_factor + sweat_factor + activity_factor

    # clamp to [0,1]
    df_local["_weighted"] = df_local["_weighted"].clip(0.0, 1.0)

    # compute distance to predicted normalized value
    df_local["_abs_diff"] = (df_local["_weighted"] - normalized_pred).abs()

    # Select top 3 (closest match), tie-breaker on higher weighted score
    top_matches = df_local.sort_values(by=["_abs_diff", "_weighted"], ascending=[True, False]).head(3)

    # Build UI display
    st.metric("Predicted Comfort Index", f"{predicted_percent:.1f} %", help="Normalized comfort score (0–100%) for the current conditions")

    st.markdown("## 🔹 Recommended Fabrics for Your Scenario")
    cols = st.columns(3)
    recommendations = []

    for i, (_, row) in enumerate(top_matches.iterrows()):
        fabric = row.get("fabric_type", "Unknown")
        weighted_pct = float(row["_weighted"] * 100.0)   # 0..100
        weighted_pct = round(weighted_pct, 1)
        explanation = generate_fabric_explanation_from_percent(fabric, weighted_pct)

        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🧵 {fabric}</h4>
                <div class="metric-value">{weighted_pct:.1f} %</div>
                <div class="metric-label">Comfort Score</div>
                <p>{explanation}</p>
            </div>
            """, unsafe_allow_html=True)

        recommendations.append({
            "Fabric": fabric,
            "Comfort Score (%)": f"{weighted_pct:.1f}",
            "Explanation": explanation
        })

    # Chart
    if recommendations:
        chart_data = pd.DataFrame(recommendations)
        chart_data["Comfort Score (%)"] = chart_data["Comfort Score (%)"].astype(float)
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Fabric", sort=None),
            y=alt.Y("Comfort Score (%)", title="Comfort (%)")
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No recommendations available—check your dataset and features.")

    # Export functions
    st.markdown("### 📤 Export Recommendation Report")
    excel_buffer = BytesIO()
    pd.DataFrame(recommendations).to_excel(excel_buffer, index=False)
    st.download_button("📊 Download Excel Report", data=excel_buffer.getvalue(),
                       file_name="fabric_recommendations.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    def generate_pdf(recs):
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Fabric Recommendation Report")
        c.setFont("Helvetica", 12)
        y = height - 100
        for rec in recs:
            c.drawString(50, y, f"Fabric: {rec['Fabric']}  |  Comfort Score: {rec['Comfort Score (%)']}%")
            y -= 20
            c.setFont("Helvetica-Oblique", 10)
            # limit explanation length per line
            text = rec['Explanation']
            c.drawString(70, y, (text[:120] + '...') if len(text) > 120 else text)
            c.setFont("Helvetica", 12)
            y -= 30
            if y < 120:
                c.showPage()
                y = height - 80
        c.save()
        pdf_buffer.seek(0)
        return pdf_buffer

    pdf_report = generate_pdf(recommendations)
    st.download_button("📄 Download PDF Report", data=pdf_report,
                       file_name="fabric_recommendations.pdf", mime="application/pdf")

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
    st.write(df_clean.describe())

    # Correlation heatmap simplified
    try:
        corr = df_clean[feature_cols + [target_col]].corr().reset_index().melt("index")
        heatmap = alt.Chart(corr).mark_rect().encode(
            x="index:O",
            y="variable:O",
            color=alt.Color("value:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["index", "variable", alt.Tooltip("value:Q", format=".2f")]
        )
        st.altair_chart(heatmap, use_container_width=True)
    except Exception:
        st.info("Correlation heatmap not available for current dataset.")

    st.markdown("#### 🧵 Top Comfort-Performing Fabrics (dataset average)")
    try:
        top_fabrics = df_clean.groupby("fabric_type")[target_col].mean().reset_index()
        top_fabrics = top_fabrics.sort_values(by=target_col, ascending=False).head(5)
        bar_chart = alt.Chart(top_fabrics).mark_bar().encode(
            x=alt.X("fabric_type", sort=None, title="Fabric"),
            y=alt.Y(target_col, title="Average Comfort Score"),
            tooltip=["fabric_type", alt.Tooltip(target_col, format=".2f")]
        )
        st.altair_chart(bar_chart, use_container_width=True)
    except Exception:
        st.info("Top fabrics chart could not be generated.")

# -------------------------------
# TAB 3: Model Performance
# -------------------------------
with tab3:
    try:
        metrics = evaluate_model(model, X_test, y_test)
        r2 = metrics["r2"]
        rmse = metrics["rmse"]
        st.metric("R² Score", f"{r2:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")
        with st.expander("ℹ️ What is R² Score?"):
            st.write(f"R² = {r2:.2f} → The model explains about {r2*100:.1f}% of the comfort variation.")
        with st.expander("ℹ️ What is RMSE?"):
            st.write(f"RMSE = {rmse:.2f} → Average prediction error.")
    except Exception as e:
        st.info(f"Model metrics not available: {e}")

# -------------------------------
# TAB 4: About
# -------------------------------
with tab4:
    st.markdown(f"""
    ## ℹ️ About {config['app']['title']}
    A professional AI system for fabric comfort and performance recommendation.
    """)
    st.markdown("---")
    st.markdown(f"🔗 **Project Repository:** [GitHub – VolandoFernando/sweatsmart-ai](https://github.com/VolandoFernando/sweatsmart-ai)")
