# sweat_smart_bot.py
import streamlit as st
from openai import OpenAI
import pandas as pd
import json

# ----------------- Page Setup -----------------
st.set_page_config(
    page_title="💬 SweatyBot – Fabric Advisor",
    page_icon="👕",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("💬 Meet SweatyBot – Your Fabric Advisor")
st.caption("Helping you find sweat-proof, eco-friendly, and comfy fabrics based on science and your needs!")

# ----------------- API Key -----------------
if "api_key" not in st.session_state:
    st.session_state.api_key = st.secrets.get("openai", {}).get("api_key", "")

api_key = st.session_state.api_key

if not api_key:
    st.warning("⚠️ Missing API key! Please add it to Streamlit Secrets as `openai.api_key`.")
    st.info("Go to Streamlit Cloud → Settings → Secrets → Add `openai.api_key = your_api_key_here`")
    st.stop()

client = OpenAI(api_key=api_key)

# ----------------- Sidebar Personalization -----------------
with st.sidebar:
    st.header("🛠 Personalize Your Recommendations")
    activity = st.selectbox("Activity Level", ["Low", "Moderate", "High"])
    climate = st.selectbox("Climate", ["Hot", "Humid", "Cold", "Moderate"])
    eco_friendly = st.checkbox("Prefer Eco-friendly fabrics?", value=True)

# ----------------- Chat History -----------------
if "messages" not in st.session_state:
    # Initial system prompt
    system_prompt = (
        f"You are SweatyBot, a helpful fabric advisor. "
        f"The user prefers '{activity}' activity, in '{climate}' climate. "
        f"Eco-friendly fabrics preference: {eco_friendly}."
    )
    st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Hi there 👋 I'm SweatyBot! Ask me anything about sweat-resistant, breathable, or eco-friendly fabrics."}
    ]

# ----------------- Display Chat Messages -----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------- User Input -----------------
if prompt := st.chat_input("Ask SweatyBot something about fabrics..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("💭 Thinking...")

        try:
            # Call OpenAI LLM
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages,
                max_tokens=500,
                temperature=0.7
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"⚠️ Oops! Something went wrong: {str(e)}"

        message_placeholder.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

        # ----------------- Optional: Try parsing JSON for fabric table -----------------
        try:
            fabrics = json.loads(reply)
            if isinstance(fabrics, list):
                df = pd.DataFrame(fabrics)
                st.markdown("### 🧵 Fabric Recommendations")
                st.table(df)
        except:
            pass

# ----------------- Collapsible Chat History -----------------
with st.expander("💬 View Full Chat History"):
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "SweatyBot"
        st.markdown(f"**{role}:** {msg['content']}")
