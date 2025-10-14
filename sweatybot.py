import streamlit as st
from openai import OpenAI

# ---------- Page setup ----------
st.set_page_config(page_title="SweatyBot – Fabric Advisor", page_icon="👕")
st.title("💬 Meet SweatyBot – Your Fabric Advisor")
st.caption("👕 Helping you find sweat-proof, eco-friendly, and comfy fabrics based on science and your needs!")

# ---------- API Key ----------
if "api_key" not in st.session_state:
    st.session_state.api_key = st.secrets.get("openai", {}).get("api_key", "")

api_key = st.session_state.api_key

if not api_key:
    st.warning("⚠️ Missing API key! Please add it to Streamlit Secrets as `openai.api_key`.")
    st.info("Go to Streamlit Cloud → Settings → Secrets → Add `openai.api_key = your_api_key_here`")
    st.stop()

# ---------- Initialize OpenAI Client ----------
client = OpenAI(api_key=api_key)

# ---------- Chat History ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi there 👋 I'm SweatyBot — your fabric advisor! Ask me anything about sweat-resistant, breathable, or eco-friendly fabrics!"
        }
    ]

# ---------- Display Chat Messages ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- User Input ----------
if prompt := st.chat_input("Ask SweatyBot something about fabrics..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("💭 Thinking...")

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # lightweight + fast
                messages=st.session_state.messages,
                max_tokens=400
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"⚠️ Oops! Something went wrong: {str(e)}"

        message_placeholder.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
