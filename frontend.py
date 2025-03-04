import streamlit as st
import requests

# Streamlit Page Configuration
st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("🧠 AI Chatbot Agents")
st.markdown("### Create and interact with AI-powered agents!")

# System Prompt Input
st.subheader("🔧 Define Your AI Agent")
system_prompt = st.text_area("System Prompt", height=80, placeholder="Describe the agent's behavior...")

# Model Provider Selection
st.subheader("🤖 Select Model Provider")
provider = st.radio("Choose Provider:", ["Groq", "Gemini"])

# Model Selection Based on Provider
MODEL_NAMES = {
    "Groq": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
    "Gemini": ["gemini-1.5-pro"],
}
selected_model = st.selectbox("Choose Model", MODEL_NAMES[provider])

# Web Search Toggle
allow_web_search = st.checkbox("🔍 Enable Web Search only with Gemini")

# User Query Input
st.subheader("💬 Ask Your Question")
user_query = st.text_area("Enter Query", height=150, placeholder="Type your question here...")

API_URL = "http://127.0.0.1:8000/chat"

# Submit Button
if st.button("🚀 Ask Agent"):
    if user_query.strip():
        st.info("Processing your request...")

        # Prepare Payload
        payload = {
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }

        # Send Request
        response = requests.post(API_URL, json=payload)

        # Handle Response
        if response.status_code == 200:
            response_data = response.json()
            if "error" in response_data:
                st.error(response_data["error"])
            else:
                st.subheader("🤖 Agent Response")
                st.markdown(f"**Final Response:** {response_data}")
        else:
            st.error("⚠️ Something went wrong. Please try again.")

# Footer
st.markdown("---")
st.caption("Powered by LangGraph 🚀")
