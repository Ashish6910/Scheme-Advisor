import os
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv

import google.generativeai as genai

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

from rag.build_vector_store import search_schemes as search_vector_store


# Environment configuration and loading 
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")



#Agent State details

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_profile: dict
    schemes_found: list
    recommendation: str
    next_action: str


# =========================
# ✅ LLM Helper function
# =========================
def call_llm(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text if response.text else ""
    except Exception as e:
        return f"Error: {str(e)}"


# ====================================
# ✅ NODE 1: Understanding user input
# ====================================
def understand_user(state: AgentState) -> AgentState:
    messages = state["messages"]
    history = "\n".join([m.content for m in messages])

    # ✅ Avoiding infinite loop
    if len(messages) > 2:
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content="Got your details. Finding best schemes for you...")
            ],
            "next_action": "search"
        }

    prompt = f"""
You are a Government Scheme Advisor.

Ask the user for missing details:
- State
- Income
- Occupation

Conversation:
{history}

Ask ONLY one question.
"""

    response = call_llm(prompt)

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "next_action": "understand"
    }


# ========================================================
# ✅ NODE 2: Searching relevant schemes as per user input
# ========================================================
def search_node(state: AgentState) -> AgentState:
    history = "\n".join([m.content for m in state["messages"]])

    prompt = f"""
Convert this into a short search query:

{history}

Return ONLY the query.
"""

    query = call_llm(prompt).strip()

    print("[SEARCH QUERY]:", query)

    results = search_vector_store(query, k=10)

    return {
        **state,
        "schemes_found": results,
        "next_action": "evaluate"
    }


# =====================================
# ✅ NODE 3: Evaluating the user input
# =====================================
def evaluate_node(state: AgentState) -> AgentState:
    schemes = state.get("schemes_found", [])
    history = "\n".join([m.content for m in state["messages"]])

    if not schemes:
        return {
            **state,
            "recommendation": "No schemes found.",
            "next_action": "recommend"
        }

    schemes_text = "\n\n".join([
        f"Scheme: {s['name']}\nBenefits: {s['benefits']}\nEligibility: {s['eligibility']}\nURL: {s['url']}"
        for s in schemes
    ])

    prompt = f"""
User profile:
{history}

Available schemes:
{schemes_text}

Suggest best 3 schemes with:
- Name
- Benefits
- Why suitable
- URL
"""

    recommendation = call_llm(prompt)

    return {
        **state,
        "recommendation": recommendation,
        "next_action": "recommend"
    }


# =============================================
# ✅ NODE 4: Final ouput displayed to the user
# =============================================
def recommend_node(state: AgentState) -> AgentState:
    return {
        **state,
        "messages": state["messages"] + [
            AIMessage(content=state["recommendation"])
        ],
        "next_action": END
    }


# =========================
# ✅ Router details
# =========================
def route(state: AgentState):
    return state.get("next_action", "understand")


# ============================================
# ✅ Building graph using LangGraph framework
# ============================================
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("understand", understand_user)
    graph.add_node("search", search_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("recommend", recommend_node)

    graph.set_entry_point("understand")

    graph.add_conditional_edges("understand", route)
    graph.add_conditional_edges("search", route)
    graph.add_conditional_edges("evaluate", route)

    graph.add_edge("recommend", END)

    return graph.compile()


app = build_graph()


# =========================
# ✅ Agent run function
# =========================
def run_agent(user_message: str, history: list):

    state = {
        "messages": history + [HumanMessage(content=user_message)],
        "user_profile": {},
        "schemes_found": [],
        "recommendation": "",
        "next_action": "understand"
    }

    result = app.invoke(state)

    ai_messages = [
        m for m in result["messages"]
        if isinstance(m, AIMessage)
    ]

    reply = ai_messages[-1].content if ai_messages else "Something went wrong."

    return {
        "type": "message",
        "text": reply
    }, result["messages"]
