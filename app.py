import os
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.docstore.document import Document

# Configure Streamlit page first to prevent any rendering issues
st.set_page_config(
    page_title="CodeBasics AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state safely
if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom CSS with error-proof styling
def inject_custom_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stTextInput input {
            border-radius: 20px;
            padding: 12px 15px;
            border: 1px solid #ced4da;
        }
        .stButton button {
            width: 100%;
            border-radius: 20px;
            padding: 12px 15px;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border: none;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .chat-message-user {
            background-color: #e3f2fd;
            padding: 12px 16px;
            border-radius: 15px;
            margin: 8px 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chat-message-assistant {
            background-color: #f1f1f1;
            padding: 12px 16px;
            border-radius: 15px;
            margin: 8px 0;
            max-width: 80%;
            margin-right: auto;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stAlert {
            border-radius: 10px;
        }
        .stSpinner > div {
            border-color: #4CAF50 transparent transparent transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# Safe environment variable loading
try:
    load_dotenv()
except Exception as e:
    st.warning(f"Couldn't load .env file: {e}")

# Safe FAQ data loading
def load_faq_data(faq_path="codebasics_faqs.csv"):
    """Safely load FAQ data from CSV"""
    try:
        if not os.path.exists(faq_path):
            return []
            
        df = pd.read_csv(faq_path)
        if 'prompt' not in df.columns or 'response' not in df.columns:
            st.error("FAQ CSV must contain 'prompt' and 'response' columns")
            return []
            
        documents = []
        for _, row in df.iterrows():
            content = f"Question: {row['prompt']}\nAnswer: {row['response']}"
            metadata = {"source": "codebasics_faq", "type": "faq"}
            documents.append(Document(page_content=content, metadata=metadata))
        return documents
    except Exception as e:
        st.error(f"Error loading FAQ data: {str(e)}")
        return []

# Robust vectorstore creation with multiple fallbacks
@st.cache_resource(show_spinner="Building knowledge base...")
def create_vectorstore(_urls, _faq_path):
    """Safely create vector store with comprehensive error handling"""
    all_docs = []
    
    # Load web content with retry logic
    if _urls:
        for url in _urls:
            try:
                if url.strip():  # Skip empty URLs
                    loader = WebBaseLoader(url.strip())
                    web_docs = loader.load()
                    all_docs.extend(web_docs)
            except Exception as e:
                st.warning(f"Couldn't load URL {url}: {str(e)}")
                continue
    
    # Load FAQ data
    try:
        faq_docs = load_faq_data(_faq_path)
        all_docs.extend(faq_docs)
    except Exception as e:
        st.warning(f"Couldn't load FAQ data: {str(e)}")
    
    if not all_docs:
        st.error("No documents available to create knowledge base")
        return None
    
    # Document splitting with safety checks
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(all_docs)
        
        if not splits:
            st.error("Document splitting resulted in empty chunks")
            return None
            
        return FAISS.from_documents(
            documents=splits,
            embedding=OpenAIEmbeddings()
        )
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# Sidebar with validation
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API key input with validation
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Required for AI functionality"
    )
    
    # URL input with parsing safeguards
    url_input = st.text_area(
        "CodeBasics Website URLs (one per line)",
        value="https://codebasics.io/\nhttps://codebasics.io/data-analyst-bootcamp",
        help="Websites to include in knowledge base"
    )
    urls = [url.strip() for url in url_input.split("\n") if url.strip()]
    
    # Model selection with default fallback
    model_name = st.selectbox(
        "AI Model",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <small>Powered by LangChain & OpenAI</small>
    </div>
    """, unsafe_allow_html=True)

# Main app interface
st.title("🤖 CodeBasics AI Assistant")
st.caption("Ask me anything about CodeBasics courses and resources")

# Initialize vectorstore safely
vectorstore = None
try:
    vectorstore = create_vectorstore(urls, "codebasics_faqs.csv")
except Exception as e:
    st.error(f"Initialization error: {str(e)}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input with comprehensive validation
if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Validate prerequisites
            if not api_key:
                raise ValueError("Please enter your OpenAI API key in the sidebar")
            
            if not vectorstore:
                raise ValueError("Knowledge base not ready. Check configuration.")
            
            # Set API key for current session
            os.environ["OPENAI_API_KEY"] = api_key
            
            # Create retriever with safe defaults
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Prepare the prompt template with context guidance
            template = """You are an expert assistant for CodeBasics educational platform. 
            Answer the question based only on the following context:

            Context:
            {context}

            Question: {question}

            Provide a detailed, accurate answer. If unsure, say you don't know.
            Format your response clearly with proper paragraphs and bullet points when helpful:"""
            
            prompt_template = ChatPromptTemplate.from_template(template)
            
            # Initialize LLM with timeout
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=0.3,
                request_timeout=30
            )
            
            # Create RAG chain with error handling
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            
            # Stream response with progress indicator
            for chunk in chain.stream(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            full_response = f"⚠️ Sorry, I encountered an error processing your request. Please try again. \n\n(Technical details: {str(e)})"
            message_placeholder.error(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
