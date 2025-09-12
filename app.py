# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from utils import (
    load_config,
    load_datasets,
    detect_features_and_target,
    train_model,
    evaluate_model,
)

# -------------------------
# Page config
# -------------------------
config = load_config()
st.set_page_config(page_title=config["app"]["title"], layout="wide", page_icon="👕")

THEME = config["app"].get("theme_color", "#1F77B4")

# -------------------------
# Top header / intro
# -------------------------
st.markdown(
    f"""
    <style>
      .title {{ font-size:28px; font-weight:700; color:{THEME}; }}
      .intro-box {{
          padding: 14px;
          border-radius: 10px;
          background: linear-gradient(135deg,#fff,#f6f9fc);
          box-shadow: 0 6px 24px rgba(16,24,40,0.06);
      }}
      .card {{ background:#fff; padding:12px; border-radius:10px; box-shadow:0 4px 12px rgba(16,24,40,0.04); }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<div class='title'>👕 {config['app']['title']}</div>", unsafe_allow_html=True)
st.markdown(f"**{config['app'].get('subtitle','Comfort & Performance Insights for Apparel Industry')}**")
st.markdown(
    """
    <div class='intro-box'>
      <strong>What this app does</strong>: recommends fabrics optimized for comfort and sweat management
      using literature + survey datasets. Use it for R&D, sourcing, or product design decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Fabric knowledge base
# -------------------------
fabric_info = {
    "cotton": "Breathable, soft, and moisture-absorbent. Good for casual and summer garments.",
    "polyester": "Durable, lightweight, quick-drying; widely used in sportswear and blends.",
    "nylon": "Strong, abrasion-resistant and elastic. Often used in activewear & outerwear.",
    "wool": "Warm, insulating, good moisture-wicking for cold weather applications.",
    "silk": "Lightweight, smooth, and thermally balanced; used in premium garments.",
    "linen": "Very breathable and cooling; ideal for hot climates but wrinkles easily.",
    "rayon": "Soft, drapey, and comfortable; used in fashion and performance blends.",
    "spandex": "Highly stretchable; usually blended to add elasticity and recovery.",
}

# -------------------------
# Load data (robust)
# -------------------------
try:
    df = load_datasets(config)
    st.success("Datasets loaded.")
except Exception as e:
    st.error(f"Failed to load datasets: {e}")
    st.stop()

# show columns for transparency (collapsible)
with st.expander("Dataset columns (preview)"):
    st.write(list(df.columns))
    st.dataframe(df.head(4))

# -------------------------
# Detect features & target
# -------------------------
feature_cols, target_col = detect_features_and_target(df, config)

if not feature_cols or target_col is None:
    st.error(
        "Could not automatically detect required features/target. "
        "Please update `config.yaml` feature keywords or confirm your dataset columns."
    )
    st.stop()

# allow user to override detected columns (safety)
st.sidebar.header("Data & Model settings (override)")
sel_features = st.sidebar.multiselect(
    "Feature columns (detected)",
    options=list(df.columns),
    default=feature_cols,
)
sel_target = st.sidebar.selectbox(
    "Target column (detected)",
    options=list(df.columns),
    index=list(df.columns).index(target_col) if target_col in df.columns else 0,
)

if len(sel_features) < 1:
    st.sidebar.error("Pick at least one feature column.")
    st.stop()

# -------------------------
# Train model (cached implicitly)
# -------------------------
try:
    model, scaler, X_test, y_test, df_clean = train_model(df, sel_features, sel_target, config)
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

# quick model evaluation summary
metrics = evaluate_model(model, X_test, y_test)

col1, col2, col3 = st.columns([1, 1, 2])
col1.metric("R²", metrics["r2"])
col2.metric("RMSE", metrics["rmse"])
col3.write(f"Training rows: **{len(df_clean):,}**  •  Features: **{len(sel_features)}**")

# -------------------------
# Recommender UI (main)
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Recommender", "📊 Insights", "🤖 Model", "ℹ️ About"])

with tab1:
    st.markdown("### Set environment & user conditions")
    with st.form("env_form"):
        c1, c2 = st.columns(2)
        with c1:
            temperature = st.slider("🌡️ Temperature (°C)", 0, 50, 28)
            humidity = st.slider("💧 Humidity (%)", 0, 100, 60)
        with c2:
            sweat_sensitivity = st.selectbox("🧍 Sweat Sensitivity", ["Low", "Medium", "High"])
            activity_intensity = st.selectbox("🏃 Activity Intensity", ["Low", "Moderate", "High"])

        # optional advanced: allow editing numeric features individually if user wants exact values
        advanced = st.checkbox("Show/override numeric feature sliders (advanced)", value=False)
        manual_feature_values = {}
        if advanced:
            st.markdown("#### Override per-feature values")
            for f in sel_features:
                if pd.api.types.is_numeric_dtype(df_clean[f]):
                    mn = float(df_clean[f].min())
                    mx = float(df_clean[f].max())
                    md = float(df_clean[f].median())
                    manual_feature_values[f] = st.slider(f"{f}", mn, mx, md, key=f"manual_{f}")

        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        # encode simple heuristic for common features; fall back to manual or median
        sweat_map = {"Low": 1, "Medium": 2, "High": 3}
        activity_map = {"Low": 1, "Moderate": 2, "High": 3}
        s_val = sweat_map[sweat_sensitivity]
        a_val = activity_map[activity_intensity]

        user_vector = []
        for f in sel_features:
            fname = f.lower()
            if "moist" in fname or "regain" in fname:
                val = s_val * 5.0
            elif "water" in fname or "absorp" in fname:
                val = 800.0 + humidity * 5.0
            elif "dry" in fname or "drying" in fname:
                val = 60.0 + a_val * 10.0
            elif "therm" in fname or "conduct" in fname or "thermal" in fname:
                val = 0.04 + (temperature - 25.0) * 0.001
            elif f in manual_feature_values:
                val = float(manual_feature_values[f])
            else:
                # fallback to median of that column
                try:
                    val = float(df_clean[f].median())
                except Exception:
                    val = 0.0
            user_vector.append(val)

        user_arr = np.array([user_vector], dtype=float)
        user_scaled = scaler.transform(user_arr)
        predicted = float(model.predict(user_scaled)[0])

        # find top matches by absolute difference on target
        df_clean["_pred_diff"] = (df_clean[sel_target].astype(float) - predicted).abs()
        top_k = int(config.get("app", {}).get("top_k", 5))
        top = df_clean.sort_values("_pred_diff").head(top_k).copy()

        # Prepare display
        st.markdown("### 🔹 Recommendations")
        cards = st.columns(min(3, len(top)))
        recommendations = []
        for i, (_, r) in enumerate(top.iterrows()):
            fabric_label = r.get("fabric_type") or r.get("material") or r.get("name") or "Unknown"
            explanation = fabric_info.get(str(fabric_label).strip().lower().capitalize(), None)
            # try lowercase lookup
            if not explanation:
                explanation = fabric_info.get(str(fabric_label).strip().lower(), "Description not available.")
            score = float(r[sel_target])

            recommendations.append(
                {
                    "Fabric": fabric_label,
                    "Comfort Score": round(score, 3),
                    "Delta": round(r["_pred_diff"], 4),
                    "Explanation": explanation,
                }
            )

            with cards[i % len(cards)]:
                st.markdown(
                    f"""
                    <div class="card">
                      <h4>🧵 {fabric_label}</h4>
                      <div style="font-size:20px;font-weight:700">{score:.3f}</div>
                      <div style="color:#6B7280">Comfort Score</div>
                      <p style="margin-top:8px">{explanation}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # show table of recommendations and allow download
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df)

        # -------------------------
        # Export Excel
        # -------------------------
        excel_buffer = BytesIO()
        rec_df.to_excel(excel_buffer, index=False)
        st.download_button(
            "⬇️ Download Excel Report",
            data=excel_buffer.getvalue(),
            file_name="fabric_recommendations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # -------------------------
        # Export PDF (simple executive report)
        # -------------------------
        def build_pdf(recs: pd.DataFrame, user_meta: dict) -> BytesIO:
            buf = BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            width, height = A4
            # Header
            c.setFont("Helvetica-Bold", 16)
            c.drawString(40, height - 50, f"{config['app']['title']} — Recommendation Report")
            c.setFont("Helvetica", 10)
            c.drawString(40, height - 68, f"Generated by app — {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
            # User scenario
            c.setFont("Helvetica-Bold", 11)
            c.drawString(40, height - 95, "Scenario:")
            c.setFont("Helvetica", 10)
            y = height - 110
            for k, v in user_meta.items():
                c.drawString(48, y, f"{k}: {v}")
                y -= 14
            y -= 6
            c.setFont("Helvetica-Bold", 11)
            c.drawString(40, y, "Top Recommendations:")
            y -= 18
            c.setFont("Helvetica", 10)
            for _, row in recs.iterrows():
                if y < 80:
                    c.showPage()
                    y = height - 60
                c.drawString(48, y, f"• {row['Fabric']} — Comfort: {row['Comfort Score']}, Δ: {row['Delta']}")
                y -= 12
                c.setFont("Helvetica-Oblique", 9)
                c.drawString(60, y, f"  {row['Explanation']}")
                y -= 16
                c.setFont("Helvetica", 10)
            c.showPage()
            c.save()
            buf.seek(0)
            return buf

        user_meta = {
            "Temperature (°C)": temperature,
            "Humidity (%)": humidity,
            "Sweat Sensitivity": sweat_sensitivity,
            "Activity Intensity": activity_intensity,
            "Predicted Comfort Score": round(predicted, 4),
        }
        pdf_buf = build_pdf(rec_df, user_meta)
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_buf,
            file_name="fabric_recommendations.pdf",
            mime="application/pdf",
        )

with tab2:
    st.markdown("### Dataset snapshot")
    st.dataframe(df_clean.head(20))
    st.markdown("#### Summary statistics")
    st.write(df_clean.describe(include="all"))

    # correlation heatmap of selected features + target
    try:
        corr_df = df_clean[sel_features + [sel_target]].corr().reset_index().melt("index")
        heat = (
            alt.Chart(corr_df)
            .mark_rect()
            .encode(x="index:O", y="variable:O", color="value:Q", tooltip=["index", "variable", "value"])
        )
        st.altair_chart(heat, use_container_width=True)
    except Exception:
        st.info("Not enough numeric columns to render heatmap.")

with tab3:
    st.markdown("### Model performance")
    st.write("Evaluation on hold-out test set:")
    st.metric("R²", metrics["r2"])
    st.metric("RMSE", metrics["rmse"])

    # feature importance
    try:
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"feature": sel_features, "importance": importances}).sort_values("importance", ascending=False)
        bar = alt.Chart(fi_df).mark_bar().encode(x="feature:N", y="importance:Q", tooltip=["feature", "importance"])
        st.altair_chart(bar, use_container_width=True)
    except Exception:
        st.info("Model does not expose feature_importances_")

with tab4:
    st.markdown("### About this tool")
    st.write(
        f"{config['app']['title']} — an industry-oriented fabric recommender for R&D & product teams. "
        "Adjust environmental & user conditions, review top fabrics and download a PDF/Excel executive report."
    )
    st.write("Built by: Volando Fernando")
