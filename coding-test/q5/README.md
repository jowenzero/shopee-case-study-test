# Food Receipt AI Platform

AI-powered platform for uploading food receipts, extracting data using OCR, and querying using natural language with LLM.

## Features

- Upload food receipts (JPG/PNG)
- Automatic data extraction using Gemini Vision API
- Structured data storage (SQLite + Vector DB)
- Natural language querying with Gemini 2.5 Flash
- LangGraph-powered agent workflow
- Semantic search capabilities

## Tech Stack

- **Frontend**: Streamlit
- **OCR**: Google Gemini 2.5 Flash (Vision API)
- **Storage**: SQLite + Custom Vector DB
- **LLM**: Google Gemini 2.5 Flash
- **Framework**: LangChain + LangGraph

## Prerequisites

- Python 3.8 or higher
- Google Gemini API Key

## Installation

### Quick Setup (Linux/Mac)

```bash
# Run the setup script
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Edit .env and add your GEMINI_API_KEY
nano .env
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Configuration

Edit `.env` file:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
DB_PATH=./data/receipts.db
VECTOR_DB_PATH=./data/vector_db.json
```

## Usage

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate

# Run the application
streamlit run app.py
```

Access the app at: http://localhost:8501

The application has multiple pages:
- **Main page**: Upload and view receipts
- **Query Receipts**: Natural language querying (accessible from sidebar)

## Project Structure

```
q5/
├── src/
│   ├── vector_db/              # Vector database from q4
│   ├── ocr_extractor.py        # OCR extraction
│   ├── database.py             # SQLite operations
│   ├── embeddings.py           # Vector embeddings
│   ├── storage_integration.py  # OCR + DB integration
│   ├── langchain_tools.py      # LangChain custom tools
│   ├── langgraph_agent.py      # LangGraph agent workflow
│   └── llm_config.py           # Gemini LLM configuration
├── pages/                      # Streamlit multi-page app
│   └── query_receipts.py       # Query receipts page
├── tests/                      # Test scripts
│   ├── test_ocr.py             # Tests all 4 receipt images
│   ├── test_database.py
│   ├── test_langchain_tools.py
│   ├── test_langgraph_agent.py
│   ├── test_imports.py
│   └── test_storage_integration.py
├── app.py                      # Main Streamlit app (Upload & View)
├── uploads/                    # Uploaded receipt images
├── data/                       # Database files
└── requirements.txt            # Python dependencies
```

## Example Queries

- "What food did I buy yesterday?"
- "Give me total expenses for food this year"
- "Where did I buy nasi goreng from last 3 months?"

## Testing

```bash
# Activate virtual environment
source venv/bin/activate

# Test package imports
python tests/test_imports.py

# Test database operations
python tests/test_database.py

# Test OCR extraction on all 4 test receipts
python tests/test_ocr.py

# Test storage integration (SQLite + Vector DB)
python tests/test_storage_integration.py

# Test LangChain tools
python tests/test_langchain_tools.py

# Test LangGraph agent workflow
python tests/test_langgraph_agent.py
```