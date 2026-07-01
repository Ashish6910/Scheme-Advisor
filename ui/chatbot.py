"""
✅ Streamlit chatbot

Run:
    streamlit run ui/chatbot.py
"""

import sys
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# ✅ Import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.scheme_agent import run_agent


# =========================
# ✅ Page configuration
# =========================
st.set_page_config(
    page_title="Indian Scheme Advisor",
    page_icon="🏛️",
    layout="centered"
)

st.markdown("## 🏛️ Indian Government Scheme Advisor")
st.markdown(
    "Built using **LangGraph + Sentence Transformers + ChromaDB + Google Gemini**"
)
st.divider()


# ================================
# ✅ Session state initialization
# ================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.graph_history = []

    welcome = (
        "🙏 **Namaste!**\n\n"
        "I'm your Government Scheme Advisor.\n\n"
        "Tell me about yourself so I can recommend the best schemes for you."
    )

    # ✅ UI message
    st.session_state.messages.append({
        "role": "assistant",
        "content": welcome
    })

    # ✅ Agent memory
    st.session_state.graph_history.append(
        AIMessage(content=welcome)
    )


# =========================
# ✅ Display chat history
# =========================
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "🏛️" if role == "assistant" else "👤"

    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])


# =========================
# ✅ User Input
# =========================
if user_input := st.chat_input("Type your message..."):

    # ✅ Add user message to UI
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # =========================
    # ✅ Agent run
    # =========================
    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("🔎 Finding relevant schemes for you..."):

            try:
                response, updated_history = run_agent(
                    user_input,
                    st.session_state.graph_history
                )

                reply = response.get("text", "No response generated.")

                # ✅ Show reply
                st.markdown(reply)

                # ✅ Update memory
                st.session_state.graph_history = updated_history

                # ✅ Add assistant reply to UI
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply
                })

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


# =========================
# ✅ Collapsable sidebar
# =========================
with st.sidebar:
    st.markdown("### 📘 About This Project")

    st.markdown("""
**Capstone Project - TCS**

### 🚨 Problem
Citizens must manually search through hundreds of schemes on myscheme.gov.in.

### ✅ Solution
This AI system:
- Understands your profile  
- Searches schemes using semantic search (RAG)  
- Recommends best matches instantly  

### ⚙️ Tech Stack
- LangGraph (Agent flow)
- ChromaDB (Vector DB)
- Sentence Transformers (Embeddings)
- Google Gemini (LLM)
- Streamlit (UI)
""")

    st.divider()

    # ✅ Reset button
    if st.button("🔄 Start Over", use_container_width=True):
        st.session_state.clear()
        st.rerun()