# sweatybot.py
import streamlit as st
import json
import time
from groq import Groq

def render():
    st.header("💬 SweatyBot – Your AI Fabric Advisor")
    st.caption("👕 Smart, scientific, and simple advice on fabrics, comfort & sweat control.")

    # --- API Key ---
    if "groq" not in st.secrets or "api_key" not in st.secrets["groq"]:
        st.error("⚠️ Missing Groq API key! Please add it under [groq] in Streamlit Secrets.")
        st.info("""
        Go to: **Settings → Secrets → Add this:**
        ```
        [groq]
        api_key = "gsk_your_groq_api_key_here"
        ```
        """)
        st.stop()

    api_key = st.secrets["groq"]["api_key"]
    client = Groq(api_key=api_key)

    # --- Sidebar personalization ---
    with st.sidebar:
        st.subheader("🧭 Personalize Chat")
        activity = st.selectbox("🏃 Activity Level", ["Low", "Moderate", "High"], index=1)
        climate = st.selectbox("🌡️ Climate", ["Hot", "Humid", "Cold", "Moderate"], index=3)
        eco = st.checkbox("🌱 Eco-Friendly Fabrics", True)
        lang = st.selectbox("🌍 Language", ["English", "Sinhala", "Tamil"], index=0)

    # --- Initialize Chat ---
    if "messages" not in st.session_state:
        prompt = (
            f"You are SweatyBot, a multilingual AI fabric advisor. "
            f"User's activity: {activity}, climate: {climate}, eco preference: {eco}. "
            f"Reply in {lang}. Keep answers short, friendly, and based on fabric science."
        )
        st.session_state.messages = [
            {"role": "system", "content": prompt},
            {"role": "assistant", "content": f"Hi 👋 I'm SweatyBot! I can chat in {lang}. Ask me about fabrics or comfort!"}
        ]

    # --- Display chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat input ---
    if user_input := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("💭 Thinking...")

            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",  # Groq's best model
                    messages=st.session_state.messages,
                    temperature=0.7,
                    max_tokens=500
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"⚠️ Error calling Groq API: {e}"

            # Typewriter effect
            full_reply = ""
            for ch in reply:
                full_reply += ch
                placeholder.markdown(full_reply)
                time.sleep(0.01)

            st.session_state.messages.append({"role": "assistant", "content": reply})

            # Show table if JSON detected
            try:
                parsed = json.loads(reply)
                if isinstance(parsed, list):
                    st.markdown("### 🧵 Fabric Recommendations")
                    st.table(parsed)
            except Exception:
                pass

    # --- View chat history ---
    with st.expander("📜 Full Chat History"):
        for msg in st.session_state.messages:
            who = "👤 You" if msg["role"] == "user" else "🤖 SweatyBot"
            st.markdown(f"**{who}:** {msg['content']}")
