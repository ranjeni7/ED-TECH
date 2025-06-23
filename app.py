import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.set_page_config(page_title="FAQ Assistant", layout="wide")
st.title("💬 Codebasics FAQ Assistant")

# Load and cache data
@st.cache_data
def load_data():
    return pd.read_csv("codebasics_faqs.csv", encoding="utf-8", errors="ignore")

df = load_data()

# Display CSV
if st.checkbox("Show FAQ Dataset"):
    st.dataframe(df)

# Preprocess data
@st.cache_data
def create_vectorizer_and_matrix(questions):
    vectorizer = TfidfVectorizer(stop_words='english')
    question_vectors = vectorizer.fit_transform(questions)
    return vectorizer, question_vectors

vectorizer, question_vectors = create_vectorizer_and_matrix(df['Question'])

# Query box
user_query = st.text_input("Ask your question here 👇")

# Process query
def get_best_answer(query):
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, question_vectors)
    best_match_index = similarity_scores.argmax()
    best_question = df.iloc[best_match_index]["Question"]
    best_answer = df.iloc[best_match_index]["Answer"]
    score = similarity_scores[0][best_match_index]
    return best_question, best_answer, score

# Result
if user_query:
    best_q, best_a, sim_score = get_best_answer(user_query)
    if sim_score > 0.2:
        st.success("✅ Best Matched FAQ:")
        st.markdown(f"**Q: {best_q}**")
        st.markdown(f"**A:** {best_a}")
        st.caption(f"Match Confidence: {sim_score:.2f}")
    else:
        st.error("❌ No relevant answer found. Please try rephrasing your question.")

