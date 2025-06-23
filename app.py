import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load CSV data
@st.cache_data
def load_data():
    df = pd.read_csv("codebasics_faqs.csv")
    return df[['prompt', 'response']]

# Build FAISS index
@st.cache_resource
def build_faiss_index(df, model):
    embeddings = model.encode(df['prompt'].tolist())
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

# Search function
def get_answer(query, df, model, index, top_k=1):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)
    result = df.iloc[indices[0][0]]
    return result['response']

# App UI
st.set_page_config(page_title="RAG FAQ Assistant", layout="wide")
st.title("🤖 RAG FAQ Assistant")
st.markdown("Ask a question related to the uploaded FAQ!")

model = load_model()
df = load_data()
index, _ = build_faiss_index(df, model)

# Sidebar - Quick Questions
st.sidebar.title("💡 Quick Questions")
for i, row in df.head(5).iterrows():
    if st.sidebar.button(row['prompt']):
        st.session_state['quick_query'] = row['prompt']

# Main Input
query = st.text_input("📩 Ask your question:", value=st.session_state.get('quick_query', ''))
if query:
    with st.spinner("Thinking..."):
        answer = get_answer(query, df, model, index)
        st.success(answer)

# Clear session state for sidebar
st.session_state['quick_query'] = ''
