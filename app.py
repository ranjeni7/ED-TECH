import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from typing import List, Tuple
from io import StringIO
from sklearn.metrics.pairwise import cosine_similarity

# Safe import for sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("Please install sentence-transformers via `pip install sentence-transformers`")
    st.stop()

# Custom CSS
st.set_page_config(page_title="Codebasics RAG Assistant", page_icon="🤖", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; }
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #1f77b4; }
    .user-message { background-color: #f0f2f6; border-left-color: #ff6b6b; }
    .assistant-message { background-color: #e8f4f8; border-left-color: #1f77b4; }
    .ai-tag { font-size: 0.8rem; color: #4CAF50; font-style: italic; }
    .warning { color: #ff9800; font-style: italic; }
    .sidebar-info { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

class RAGAssistant:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.faq_data = self.load_faq_data()
        self.embeddings = self.model.encode(
            ["Question: " + q + " Answer: " + a for q, a in zip(self.faq_data.prompt, self.faq_data.response)]
        )

    def load_faq_data(self):
        faq_text = """prompt,response
I have never done programming in my life. Can I take this bootcamp?,"Yes, this is the perfect bootcamp for beginners..."
Why should I trust Codebasics?,"Till now 9000 + learners have benefitted..."
Is there any prerequisite for taking this bootcamp?,"Our bootcamp is specifically designed for beginners..."
What datasets are used in this bootcamp?,"The datasets mimic real business problems..."
I'm not sure if this bootcamp is good enough. What can I do?,"We got you covered. Watch our YouTube videos first..."
How can I contact the instructors?,"Join our Discord community for support..."
What if I don't like this bootcamp?,"We offer a 100% refund as per policy..."
Does this bootcamp have lifetime access?,"Yes"
What is the duration of this bootcamp?,"Complete in 3 months with 2-3 hours/day..."
Can I attend this bootcamp while working full time?,"Yes. This bootcamp is self-paced..."
"""
        return pd.read_csv(StringIO(faq_text))

    def search_similar_questions(self, query: str, top_k: int = 3):
        query_embedding = self.model.encode([query])
        sims = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = sims.argsort()[::-1][:top_k]
        return [(i, sims[i], self.faq_data.prompt[i], self.faq_data.response[i]) for i in top_indices if sims[i] > 0.3]

    def generate_llm_response(self, prompt: str, api_key: str):
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=20)
            if response.status_code == 503:
                wait_time = response.json().get('estimated_time', 30)
                time.sleep(wait_time)
                response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=20)
            if response.status_code == 200:
                return response.json()[0]['generated_text']
            return None
        except Exception:
            return None

    def generate_response(self, user_question: str, similar_questions: List[Tuple], api_key: str = None):
        if not similar_questions:
            return "I couldn't find relevant information. Please contact support."
        best_answer = similar_questions[0][3]
        if not api_key:
            return best_answer
        context = "\n".join([f"Question: {q}\nAnswer: {a}" for _, _, q, a in similar_questions])
        prompt = f"""Improve this answer for the question below using the provided context.

Original Question: {user_question}
Context:
{context}

Enhanced Answer:"""
        llm_response = self.generate_llm_response(prompt, api_key)
        if llm_response:
            return f"{llm_response}\n\n<small class='ai-tag'>AI-enhanced answer</small>"
        return f"{best_answer}\n\n<small class='warning'>Original answer (AI service unavailable)</small>"

def main():
    st.markdown('<h1 class="main-header">🤖 Codebasics RAG+LLM Assistant</h1>', unsafe_allow_html=True)

    if 'assistant' not in st.session_state:
        st.session_state.assistant = RAGAssistant()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    with st.sidebar:
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Configuration")
        hf_api_key = st.text_input("🔑 HuggingFace API Key (optional)", type="password")
        top_k = st.slider("Number of similar questions", 1, 5, 3)
        st.metric("Total FAQ Entries", len(st.session_state.assistant.faq_data))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 🚀 Quick Questions")
        quick_questions = [
            "Can I take this without programming experience?",
            "What are the course prerequisites?",
            "How long is the bootcamp?",
            "Is there job assistance?"
        ]
        for i, q in enumerate(quick_questions):
            if st.button(f"✨ {q}", key=f"quick_q_{i}"):
                st.session_state.user_input = q
                st.rerun()

        if st.button("🧹 Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.user_input = ""
            st.rerun()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 💬 Chat with Assistant")
        user_question = st.text_input("Ask about our bootcamp:", key="main_input", value=st.session_state.user_input)
        if st.button("🚀 Get Answer", type="primary") and user_question:
            st.session_state.user_input = user_question
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.spinner("🔍 Searching knowledge base..."):
                similar = st.session_state.assistant.search_similar_questions(user_question, top_k)
            with st.spinner("🧠 Enhancing with AI..." if hf_api_key else "📖 Preparing answer..."):
                reply = st.session_state.assistant.generate_response(user_question, similar, hf_api_key if hf_api_key else None)
            st.session_state.messages.append({"role": "assistant", "content": reply, "similar_questions": similar})

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)
                if "similar_questions" in msg:
                    with st.expander("📚 Source Questions"):
                        for i, (_, score, question, _) in enumerate(msg["similar_questions"], 1):
                            st.markdown(f"**{i}.** {question}  \n_Relevance: {score:.2f}_")

if __name__ == "__main__":
    main()
