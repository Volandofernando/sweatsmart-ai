# sweatybot.py
import streamlit as st
import json
try:
    from openai import OpenAI
except Exception:
    # fallback to openai package older style
    import openai as _openai
    OpenAI = None

def render():
    # Note: app.py already called set_page_config
    st.header("💬 SweatyBot – Fabric Advisor")
    st.caption("Ask about fabrics, moisture, thermal comfort, and get plain-language answers.")

    # Check secret
    if "openai" not in st.secrets or "api_key" not in st.secrets["openai"]:
        st.warning("⚠️ Missing OpenAI API key in Streamlit Secrets. Add under [openai] api_key = ...")
        return

    api_key = st.secrets["openai"]["api_key"]

    # Create client robustly
    client = None
    try:
        if OpenAI is not None:
            client = OpenAI(api_key=api_key)
        else:
            _openai.api_key = api_key
            client = _openai
    except Exception as e:
        st.error(f"Failed to create OpenAI client: {e}")
        return

    # Sidebar personalization
    with st.sidebar:
        st.subheader("Personalize")
        activity = st.selectbox("Activity level", ["Low","Moderate","High"], index=1)
        climate = st.selectbox("Climate", ["Moderate","Hot","Humid","Cold"], index=0)
        eco = st.checkbox("Prefer eco-friendly fabrics?", value=True)

    if "messages" not in st.session_state:
        system_prompt = (
            f"You are SweatyBot, an expert fabric advisor. User activity: {activity}, climate: {climate}, "
            f"eco preference: {eco}. Give short, actionable, non-technical answers and offer suggestions."
        )
        st.session_state.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": "Hi 👋 I'm SweatyBot. Ask me anything about fabrics or comfort!"}
        ]

    # Show chat
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

    # Input
    if user_input := st.chat_input("Ask SweatyBot about fabrics..."):
        st.session_state.messages.append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("💭 Thinking...")

            try:
                # New style client (OpenAI()) has chat.completions.create
                if OpenAI is not None:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=st.session_state.messages,
                        max_tokens=500,
                        temperature=0.7
                    )
                    reply = response.choices[0].message.content
                else:
                    # fallback to openai.ChatCompletion.create
                    resp = _openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=st.session_state.messages,
                        max_tokens=500,
                        temperature=0.7
                    )
                    reply = resp.choices[0].message["content"]
            except Exception as e:
                reply = f"⚠️ Error calling OpenAI API: {e}"

            placeholder.markdown(reply)
            st.session_state.messages.append({"role":"assistant","content":reply})

            # Try parse JSON if model returned structured recommendation
            try:
                parsed = json.loads(reply)
                if isinstance(parsed, list):
                    st.markdown("### Structured Recommendations")
                    st.table(parsed)
            except Exception:
                pass

    with st.expander("💬 Conversation history"):
        for m in st.session_state.messages:
            who = "You" if m["role"]=="user" else "SweatyBot"
            st.markdown(f"**{who}:** {m['content']}")
