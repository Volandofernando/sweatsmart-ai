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
    and **comfortable** fabrics based on science and your needs.  
    Type your question below and let’s get comfy together 😅.
    """
)
st.divider()

# ------------------ LOAD API KEY ------------------
# 🔐 Store your API key safely in Streamlit Cloud under "Secrets" → "openai"
try:
    client = OpenAI(api_key=st.secrets["openai"]["rswA"])
except Exception:
    st.error("⚠️ Missing API key! Please add it to Streamlit Secrets under `openai.api_key`.")
    st.stop()

# ------------------ CHAT HISTORY ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hey there 👋 I'm SweatyBot! Ask me about fabrics, sweat resistance, or comfort."}
    ]

# ------------------ DISPLAY CHAT ------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ USER INPUT ------------------
user_input = st.chat_input("Ask me something about fabrics... 🧵")

if user_input:
    # Show user input
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # ------------------ BOT REPLY ------------------
    with st.spinner("SweatyBot is thinking... 💭"):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.chat_history
            )
            reply = response.choices[0].message.content.strip()

            # Display and store the reply
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        except Exception as e:
            st.error(f"❌ Oops! SweatyBot had a hiccup: {e}")
