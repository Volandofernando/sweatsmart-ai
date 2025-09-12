# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="👕 Fabric Comfort AI Recommender",
    layout="wide"
)

st.title("👕 Fabric Comfort AI Recommender")
st.markdown(
    """
    ### Comfort & Performance Insights for Apparel Industry  
    Trusted by textile R&D, apparel design, and sportswear innovation teams.  
    **Powered by AI trained on fabric properties + real-world comfort data.**
    """
)

# -------------------------------
# LOAD DATASETS FROM GITHUB RAW
# -------------------------------
@st.cache_data
def load_datasets():
    try:
        df_materials = pd.read_excel(
            "https://github.com/Volandofernando/Material-Literature-data-/raw/main/Dataset.xlsx"
        )
        df_responses = pd.read_excel(
            "https://github.com/Volandofernando/REAL-TIME-Dataset/raw/main/IT%20Innovation%20in%20Fabric%20Industry%20%20(Responses).xlsx"
        )
        return df_materials, df_responses
    except Exception as e:
        st.error(f"❌ Failed to load datasets: {e}")
        return None, None

df_materials, df_responses = load_datasets()

if df_materials is not None and df_responses is not None:
    st.success("✅ Datasets loaded successfully!")
    st.write("**Materials Dataset Preview:**", df_materials.head())
    st.write("**Industry Responses Preview:**", df_responses.head())
else:
    st.stop()

# -------------------------------
# FEATURE SELECTION
# -------------------------------
numeric_features = df_materials.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_features:
    st.error("❌ No numeric features found in dataset for recommendation system.")
    st.stop()

# -------------------------------
# USER INPUTS (Sidebar Controls)
# -------------------------------
st.sidebar.header("🎛️ Adjust Your Conditions")

user_input = {}
for feature in numeric_features:
    min_val, max_val = (
        df_materials[feature].min(),
        df_materials[feature].max(),
    )
    user_input[feature] = st.sidebar.slider(
        feature,
        float(min_val),
        float(max_val),
        float(np.median(df_materials[feature])),
    )

user_input_df = pd.DataFrame([user_input])

# -------------------------------
# MODEL: Nearest Neighbors
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_materials[numeric_features])
user_scaled = scaler.transform(user_input_df)

model = NearestNeighbors(n_neighbors=5, metric="euclidean")
model.fit(X_scaled)
distances, indices = model.kneighbors(user_scaled)

# -------------------------------
# DISPLAY RECOMMENDATIONS
# -------------------------------
st.subheader("🔍 Recommended Fabrics")
recommended_fabrics = df_materials.iloc[indices[0]].copy()
recommended_fabrics["Similarity Score"] = 1 / (1 + distances[0])

st.dataframe(recommended_fabrics)

# -------------------------------
# VISUAL INSIGHTS
# -------------------------------
import altair as alt

st.subheader("📊 Comfort & Performance Insights")
for feature in numeric_features[:3]:  # show first 3 features
    chart = (
        alt.Chart(recommended_fabrics)
        .mark_bar()
        .encode(
            x=alt.X("Similarity Score", title="Similarity"),
            y=alt.Y(feature, title=feature),
            color="Similarity Score",
            tooltip=["Similarity Score", feature],
        )
    )
    st.altair_chart(chart, use_container_width=True)
