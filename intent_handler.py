# intent_handler.py

INTENT_MAP = {
    "thanks": ["thank you", "thanks", "shukriya", "dhanyavaad", "thank u", "okay thanks"],
    "greeting": ["hi", "hello", "namaste", "hey", "good morning", "good evening", "hii"],
    "farewell": ["bye", "goodbye", "chalte hain", "milte hain", "alvida", "byy"],
    "who_are_you": ["tu kaun hai", "who are you", "apka naam kya hai", "tum kaun ho", "what is your name"],
    "okay": ["ok", "okay", "theek hai", "thik hai", "fine", "hmm"]
}

def detect_intent(user_input):
    user_input = user_input.lower()
    for intent, keywords in INTENT_MAP.items():
        for keyword in keywords:
            if keyword in user_input:
                return intent
    return "general"

def get_intent_response(intent):
    if intent == "thanks":
        return "😊 You're welcome! Let me know if you need anything else."
    elif intent == "greeting":
        return "👋 Hello! I'm the official chatbot for Dotsquares. How can I assist you today?"
    elif intent == "farewell":
        return "👋 Goodbye! Have a great day ahead 😊"
    elif intent == "who_are_you":
        return "🤖 I am an AI chatbot developed to assist you with information about Dotsquares and its services."
    elif intent == "okay":
        return "✅ Alright! Let me know if you have any questions."
    return None
# ✅ Add at the end of intent_handler.py

import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("facts.json", "r", encoding="utf-8") as f:
    facts = json.load(f)

def get_faq_answer(user_query):
    questions = [item["question"] for item in facts]
    embeddings = model.encode(questions, convert_to_tensor=True)
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

    best_score, best_idx = scores.max().item(), scores.argmax().item()

    if best_score > 0.6:
        return facts[best_idx]["answer"]
    return None