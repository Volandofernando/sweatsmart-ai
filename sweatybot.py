import streamlit as st
import pandas as pd
import json
from openai import OpenAI

# ------------------------------
# 🔧 PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="💬 SweatyBot – Fabric Advisor",
    page_icon="👕",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("💬 Meet SweatyBot – Your Fabric Advisor")
st.caption("👕 Helping you find sweat-proof, eco-friendly, and comfy fabrics based on science and your needs!")

# ------------------------------
# 🔑 API KEY HANDLING
# ------------------------------
if "api_key" not in st.session_state:
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        st.session_state.api_key = st.secrets["openai"]["api_key"]
    else:
        st.warning("⚠️ Missing API key! Please add it to Streamlit Secrets.")
        st.info("""
        👉 Go to Streamlit Cloud → ⚙️ Settings → Secrets → Add this:
        ```
        [openai]
        api_key = "sk-proj-your_key_here"
        ```
        """)
        st.stop()


client = OpenAI(api_key=st.session_state.api_key)

# ------------------------------
# 🧵 SIDEBAR PERSONALIZATION
# ------------------------------
with st.sidebar:
    st.header("🧭 Personalize Your Preferences")
    activity = st.selectbox("🏃 Activity Level", ["Low", "Moderate", "High"])
    climate = st.selectbox("🌡️ Climate Type", ["Hot", "Humid", "Cold", "Moderate"])
    eco_friendly = st.checkbox("🌱 Prefer Eco-friendly Fabrics?", value=True)
    st.markdown("---")
    st.markdown("💡 **Tip:** Adjust these before chatting for more personalized recommendations!")

# ------------------------------
# 🧠 SESSION STATE MANAGEMENT
# ------------------------------
if "messages" not in st.session_state:
    system_prompt = (
        f"You are SweatyBot, an expert fabric advisor. "
        f"The user prefers '{activity}' activity, in a '{climate}' climate. "
        f"Eco-friendly fabrics preference: {eco_friendly}. "
        f"Always give scientific but simple explanations about fabrics — their breathability, moisture control, and comfort."
    )
    st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Hi 👋 I'm SweatyBot! Ask me anything about sweat-resistant, breathable, or eco-friendly fabrics."}
    ]

# ------------------------------
# 💬 DISPLAY CHAT MESSAGES
# ------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------
# ✏️ USER INPUT & RESPONSE
# ------------------------------
if user_input := st.chat_input("Ask me about fabrics, comfort, or recommendations..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant placeholder
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("💭 Thinking...")

        try:
            # API call
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages,
                temperature=0.7,
                max_tokens=500,
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"⚠️ Error: {str(e)}"
        
        placeholder.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

        # Optional JSON Table Rendering
        try:
            fabrics = json.loads(reply)
            if isinstance(fabrics, list):
                df = pd.DataFrame(fabrics)
                st.markdown("### 🧵 Recommended Fabrics")
                st.table(df)
        except Exception:
            pass

# ------------------------------
# 📜 CHAT HISTORY
# ------------------------------
with st.expander("💬 View Full Chat History"):
    for msg in st.session_state.messages:
        role = "👤 You" if msg["role"] == "user" else "🤖 SweatyBot"
        st.markdown(f"**{role}:** {msg['content']}")
