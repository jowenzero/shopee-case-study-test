import streamlit as st
import os
from dotenv import load_dotenv

from src.storage_integration import StorageIntegration
from src.langgraph_agent import ReceiptQueryAgent
from src.database import DatabaseManager

load_dotenv()

st.set_page_config(
    page_title="Query Receipts - Food Receipt AI",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def initialize_query_components():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        st.error("GEMINI_API_KEY not found in .env file")
        st.stop()

    storage = StorageIntegration(
        db_path="./data/receipts.db",
        vector_db_path="./data/vector_db.json"
    )
    db = DatabaseManager(db_path="./data/receipts.db")
    agent = ReceiptQueryAgent(db=db, storage=storage)
    return storage, db, agent

storage_integration, db_manager, query_agent = initialize_query_components()

st.title("Query Receipts")
st.markdown("Ask questions about your receipts in natural language")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.markdown("### Example Queries")
example_cols = st.columns(2)
with example_cols[0]:
    st.markdown("- What did I buy on 2023-04-15?")
    st.markdown("- Show me my most expensive purchase")
with example_cols[1]:
    st.markdown("- Give me total expenses for 2020")
    st.markdown("- Where did I buy chicken?")

st.markdown("---")

query_input = st.text_input(
    "Ask a question about your receipts:",
    placeholder="e.g., What food did I buy yesterday?",
    key="query_input"
)

col1, col2 = st.columns([1, 5])
with col1:
    submit_query = st.button("Ask", type="primary", key="ask_button")
with col2:
    clear_history = st.button("Clear History")

if clear_history:
    st.session_state.chat_history = []
    st.rerun()

if submit_query and query_input:
    with st.spinner("Thinking..."):
        try:
            result = query_agent.query(query_input)

            st.session_state.chat_history.append({
                'query': query_input,
                'intent': result['intent'],
                'response': result['response']
            })

        except Exception as e:
            st.error(f"Error processing query: {e}")

st.markdown("---")

if st.session_state.chat_history:
    st.markdown("### Conversation History")

    for idx, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown(f"**You:** {chat['query']}")
            st.markdown(f"*Intent: {chat['intent']}*")
            st.markdown(f"**Assistant:** {chat['response']}")

            if idx < len(st.session_state.chat_history) - 1:
                st.markdown("---")
else:
    st.info("No queries yet. Ask a question to get started!")

st.sidebar.title("About")
st.sidebar.info(
    """
    This AI-powered platform allows you to:
    - Upload food receipt images
    - Extract data using Gemini Vision API
    - Store receipts in SQLite and Vector DB
    - Query receipts using natural language
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack:**")
st.sidebar.markdown("- Gemini 2.5 Flash (Vision & LLM)")
st.sidebar.markdown("- LangChain + LangGraph")
st.sidebar.markdown("- SQLite + Vector DB")
st.sidebar.markdown("- Sentence Transformers")
st.sidebar.markdown("- Streamlit")
