import os
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="CodeBasics AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
def inject_custom_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
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
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
        .sidebar .sidebar-content .block-container {
            color: white;
        }
        .reportview-container .markdown-text-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .logo {
            text-align: center;
            margin-bottom: 20px;
        }
        .logo img {
            max-width: 80%;
            border-radius: 10px;
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
        .stMarkdown {
            line-height: 1.6;
        }
        .stSpinner > div {
            border-color: #4CAF50 transparent transparent transparent !important;
        }
        .stAlert {
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load FAQ data
def load_faq_data(faq_path):
    """Load FAQ data from CSV and convert to LangChain documents"""
    try:
        df = pd.read_csv(faq_path)
        documents = []
        for _, row in df.iterrows():
            content = f"Question: {row['prompt']}\nAnswer: {row['response']}"
            metadata = {"source": "codebasics_faq", "type": "faq"}
            documents.append(Document(page_content=content, metadata=metadata))
        return documents
    except Exception as e:
        st.error(f"Error loading FAQ data: {str(e)}")
        return []

# Create vector store with caching
@st.cache_resource(show_spinner=False)
def create_vectorstore(urls, faq_path):
    with st.spinner("Loading and indexing knowledge base..."):
        # Load web data
        web_docs = []
        if urls:
            try:
                loader = WebBaseLoader(urls)
                web_docs = loader.load()
            except Exception as e:
                st.error(f"Error loading web content: {str(e)}")
        
        # Load FAQ data
        faq_docs = []
        if faq_path and os.path.exists(faq_path):
            faq_docs = load_faq_data(faq_path)
        
        # Combine all documents
        all_docs = web_docs + faq_docs
        
        if not all_docs:
            st.error("No documents available to create knowledge base")
            return None
        
        # Split and create vectorstore
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(all_docs)
        
        try:
            vectorstore = FAISS.from_documents(
                documents=splits,
                embedding=OpenAIEmbeddings()
            )
            return vectorstore
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None

# Sidebar configuration
with st.sidebar:
    st.markdown("""
    <div class="logo">
        <h2>CodeBasics AI</h2>
        <p>Your intelligent learning assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Configuration")
    
    # Input for OpenAI API key
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Get your API key from https://platform.openai.com/account/api-keys"
    )
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Input for CodeBasics URLs
    urls = st.text_area(
        "CodeBasics Website URLs (one per line)",
        value="https://codebasics.io/\nhttps://codebasics.io/data-analyst-bootcamp\nhttps://codebasics.io/data-science-bootcamp",
        help="Add relevant CodeBasics website URLs to include in the knowledge base"
    ).split("\n")
    
    # FAQ file uploader
    faq_path = "codebasics_faqs.csv"
    if not os.path.exists(faq_path):
        st.warning("FAQ file not found. Using default knowledge base only.")
    
    # Model selection
    model_name = st.selectbox(
        "Select LLM Model",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0,
        help="More powerful models may provide better answers but cost more"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <small>Powered by LangChain & OpenAI</small>
    </div>
    """, unsafe_allow_html=True)

# Main app interface
st.title("🤖 CodeBasics AI Assistant")
st.caption("Ask me anything about CodeBasics courses, bootcamps, or learning resources")

# Initialize vector store
vectorstore = create_vectorstore(urls, faq_path)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar")
                st.stop()
            
            if not vectorstore:
                st.error("Knowledge base not properly initialized. Check configuration.")
                st.stop()
            
            # Search for relevant documents
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(prompt)
            
            # Prepare the prompt template
            template = """You are a helpful assistant for CodeBasics educational platform. 
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer in a helpful, friendly tone with proper formatting:"""
            prompt_template = ChatPromptTemplate.from_template(template)
            
            # Initialize LLM
            llm = ChatOpenAI(model_name=model_name, temperature=0.3)
            
            # Create RAG chain
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            
            # Stream the response
            for chunk in chain.stream(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            full_response = f"Sorry, I encountered an error: {str(e)}"
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
