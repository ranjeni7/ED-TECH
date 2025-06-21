import streamlit as st
import pandas as pd
import numpy as np
##from sentence_transformers import SentenceTransformer
#import faiss
import requests
from typing import List, Tuple
from io import StringIO
import time

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
        self.embeddings_model = None
        self.faiss_index = None
        self.faq_data = None
        self.embeddings = None
        
    def load_embedding_model(self):
        return SentenceTransformer('all-MiniLM-L6-v2')
    
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
        self.faq_data = pd.read_csv(StringIO(faq_text))
    
    def create_embeddings(self):
        self.embeddings_model = self.load_embedding_model()
        texts = [f"Question: {row['prompt']} Answer: {row['response']}" 
                for _, row in self.faq_data.iterrows()]
        self.embeddings = self.embeddings_model.encode(texts)
        
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings.astype('float32'))
    
    def search_similar_questions(self, query: str, top_k: int = 3):
        query_embedding = self.embeddings_model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        return [(idx, score, self.faq_data.iloc[idx]['prompt'], self.faq_data.iloc[idx]['response']) 
                for idx, score in zip(indices[0], scores[0]) if score > 0.3]

    def generate_llm_response(self, prompt: str, api_key: str):
        """Improved LLM integration with retry logic"""
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"  # More reliable model
        headers = {"Authorization": f"Bearer {api_key}"}
        
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=20)
            
            # Handle model loading case
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
        
        # Without API key, return basic RAG response
        if not api_key:
            return best_answer
        
        # With API key, enhance with LLM
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
    
    # Initialize assistant
    if 'assistant' not in st.session_state:
        assistant = RAGAssistant()
        assistant.load_faq_data()
        assistant.create_embeddings()
        st.session_state.assistant = assistant
    
    with st.sidebar:
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Configuration")
        st.markdown('</div>', unsafe_allow_html=True)
        
        hf_api_key = st.text_input(
            "🔑 HuggingFace API Key (optional)",
            type="password",
            help="Get free key from https://huggingface.co/settings/tokens"
        )
        
        st.markdown("### 🔍 RAG Settings")
        top_k = st.slider("Number of similar questions", 1, 5, 3)
        
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### 📊 Knowledge Base Stats")
        st.metric("Total FAQ Entries", len(st.session_state.assistant.faq_data))
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Chat with Assistant")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        user_question = st.text_input(
            "Ask about our bootcamp:",
            placeholder="e.g., Can I take this without programming experience?",
            key="user_input"
        )
        
        if st.button("🚀 Get Answer", type="primary") and user_question:
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            with st.spinner("🔍 Searching knowledge base..."):
                similar_questions = st.session_state.assistant.search_similar_questions(user_question, top_k)
            
            with st.spinner("🧠 Enhancing with AI..." if hf_api_key else "📖 Preparing answer..."):
                response = st.session_state.assistant.generate_response(
                    user_question, 
                    similar_questions, 
                    hf_api_key if hf_api_key else None
                )
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "similar_questions": similar_questions
            })
        
        # Display conversation
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
                
                if "similar_questions" in message and message["similar_questions"]:
                    with st.expander("📚 Source Questions"):
                        for i, (_, score, question, _) in enumerate(message["similar_questions"], 1):
                            st.markdown(f"**{i}.** {question}")
                            st.markdown(f'<div class="confidence-score">Relevance: {score:.2f}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🚀 Quick Questions")
        samples = [
            "Can I take this without programming experience?",
            "What are the course prerequisites?",
            "How long is the bootcamp?",
            "Is there job assistance?"
        ]
        for q in samples:
            if st.button(f"✨ {q}"):
                st.session_state.user_input = q
                st.rerun()
        
        if st.button("🧹 Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
