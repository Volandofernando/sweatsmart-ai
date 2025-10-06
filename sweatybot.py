import streamlit as st
from openai import OpenAI

# --- Title & Intro ---
st.set_page_config(page_title="SweatyBot 💬", page_icon="🤖")
st.title("💬 Meet SweatyBot – Your Fabric Advisor")
st.caption("Ask anything about sweat-proof fabrics, comfort materials, or eco-textiles!")

# --- Load API Key ---
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# --- Session History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hey there 👋 I'm SweatyBot! Ask me about fabrics, sweat resistance, or comfort."}
    ]

# --- Chat Interface ---
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask me something...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # --- Send to OpenAI ---
    with st.spinner("SweatyBot is thinking... 💭"):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.chat_history
        )
        reply = response.choices[0].message.content

    # --- Display Response ---
    st.chat_message("assistant").write(reply)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
