from flask import Flask, request, jsonify   #type: ignore
from flask_cors import CORS                #type: ignore
from intent_handler import detect_intent, get_intent_response, get_faq_answer
import psycopg2   #type: ignore
from psycopg2.extras import RealDictCursor  #type: ignore
import requests    #type: ignore

from langgraph.graph import StateGraph, END      #type:ignore
from langchain_core.runnables import RunnableLambda   #type: ignore
from typing import TypedDict, Literal

# ✅ Define State Schema
class ChatState(TypedDict):
    message: str
    answer: str
    next: Literal["check_intent", "check_db", "check_faq", "llm", "end"]

# ✅ Flask app
app = Flask(__name__)
CORS(app)

# ✅ OpenRouter Config
OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "mistralai/mistral-7b-instruct"

# ✅ PostgreSQL Config
DB_CONFIG = {
    "host": "localhost",
    "database": "dknmu_user",
    "user": "postgres",
    "password": "123456789",
    "port": 5432
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def search_database(query):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT title, url, content FROM website_pages
            WHERE content ILIKE %s
            ORDER BY LENGTH(content) DESC
            LIMIT 1
        """, (f"%{query}%",))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result
    except:
        return None

# ✅ LangGraph Node Functions
def handle_intent(state: ChatState) -> ChatState:
    intent = detect_intent(state["message"])
    intent_response = get_intent_response(intent)
    if intent_response:
        return {"answer": intent_response, "next": "end"}
    return {"message": state["message"], "next": "check_db"}

def handle_database(state: ChatState) -> ChatState:
    db_result = search_database(state["message"])
    if db_result:
        ans = db_result['content'].strip()
        if db_result['url']:
            ans += f"\n\n🔗 [Click here]({db_result['url']})"
        return {"answer": ans, "next": "end"}
    return {"message": state["message"], "next": "check_faq"}

def handle_faq(state: ChatState) -> ChatState:
    faq = get_faq_answer(state["message"])
    if faq:
        return {"answer": faq, "next": "end"}
    return {"message": state["message"], "next": "llm"}

def handle_llm(state: ChatState) -> ChatState:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for Dotsquares company. Reply in a helpful and polite tone."},
            {"role": "user", "content": state["message"]}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=(10, 60))
        if response.status_code == 200:
            result = response.json()
            reply = result["choices"][0]["message"]["content"]
            return {"answer": reply, "next": "end"}
        else:
            return {"answer": "❌ LLM Error", "next": "end"}
    except:
        return {"answer": "❌ Server Error", "next": "end"}

# ✅ LangGraph Setup
workflow = StateGraph(ChatState)

workflow.add_node("check_intent", RunnableLambda(handle_intent))
workflow.add_node("check_db", RunnableLambda(handle_database))
workflow.add_node("check_faq", RunnableLambda(handle_faq))
workflow.add_node("llm", RunnableLambda(handle_llm))

workflow.set_entry_point("check_intent")
workflow.add_conditional_edges("check_intent", lambda x: x["next"], {
    "check_db": "check_db",
    "end": END
})
workflow.add_conditional_edges("check_db", lambda x: x["next"], {
    "check_faq": "check_faq",
    "end": END
})
workflow.add_conditional_edges("check_faq", lambda x: x["next"], {
    "llm": "llm",
    "end": END
})
workflow.add_conditional_edges("llm", lambda x: x["next"], {
    "end": END
})

graph = workflow.compile()

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip().lower()
    if not user_message:
        return jsonify({"answer": "❌ Please enter a message."}), 400

    result = graph.invoke({"message": user_message})
    return jsonify({"answer": result["answer"]})

if __name__ == "__main__":
    print("✅ LangGraph-based Dotsquares Chatbot Running...")
    app.run(debug=True, port=5000)
