import streamlit as st
from openai import OpenAI

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="SweatyBot 💬",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------ HEADER ------------------
st.title("💬 Meet SweatyBot – Your Fabric Advisor")

st.markdown(
    """
    👕 **SweatyBot** helps you discover the best **sweat-proof**, **eco-friendly**,  
    and **comfortable fabrics** based on science and your needs.  
    Type your question below and let’s get comfy together 😅.
    """
)
st.divider()

# ------------------ LOAD OPENAI API KEY ------------------
try:
    # ✅ Load API key from Streamlit Secrets
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])
except Exception as e:
    st.error("⚠️ Missing or invalid API key! Please add it to Streamlit Secrets under `[openai].api_key`.")
    st.stop()

# ------------------ CHAT HISTORY ------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey there 👋 I'm SweatyBot! Ask me about fabrics, sweat resistance, or comfort."}
    ]

# ------------------ DISPLAY CHAT ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ USER INPUT ------------------
if prompt := st.chat_input("Ask me something about fabrics... 🧵"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # ------------------ BOT RESPONSE ------------------
    with st.chat_message("assistant"):
        with st.spinner("SweatyBot is thinking... 💭"):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # ✅ Better + cheaper than gpt-3.5-turbo
                    messages=st.session_state.messages,
                    temperature=0.7
                )
                reply = response.choices[0].message.content.strip()

            except Exception as e:
                reply = f"❌ Oops! SweatyBot had a hiccup: {e}"

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("🤖 Powered by OpenAI | Created by Volando Fernando for SweatSmart AI Fabrics")
