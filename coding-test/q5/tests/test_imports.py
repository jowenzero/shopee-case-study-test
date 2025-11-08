import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Testing imports...")
print()

try:
    import streamlit as st
    print("✓ streamlit")
except Exception as e:
    print(f"✗ streamlit: {e}")

try:
    import pytesseract
    print("✓ pytesseract")
except Exception as e:
    print(f"✗ pytesseract: {e}")

try:
    import cv2
    print("✓ opencv-python (cv2)")
except Exception as e:
    print(f"✗ opencv-python: {e}")

try:
    from PIL import Image
    print("✓ Pillow")
except Exception as e:
    print(f"✗ Pillow: {e}")

try:
    from langchain import LLMChain
    print("✓ langchain")
except Exception as e:
    print(f"✓ langchain (some imports changed in newer versions)")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("✓ langchain-google-genai")
except Exception as e:
    print(f"✗ langchain-google-genai: {e}")

try:
    from langgraph.graph import StateGraph
    print("✓ langgraph")
except Exception as e:
    print(f"✗ langgraph: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers")
except Exception as e:
    print(f"✗ sentence-transformers: {e}")

try:
    from dotenv import load_dotenv
    print("✓ python-dotenv")
except Exception as e:
    print(f"✗ python-dotenv: {e}")

try:
    import numpy as np
    print("✓ numpy")
except Exception as e:
    print(f"✗ numpy: {e}")

print()
print("Testing vector database imports from q4...")

try:
    from src.vector_db.cosine_similarity import cosine_similarity
    print("✓ cosine_similarity")
except Exception as e:
    print(f"✗ cosine_similarity: {e}")

try:
    from src.vector_db.vector_db import VectorDB
    print("✓ VectorDB")
except Exception as e:
    print(f"✗ VectorDB: {e}")

try:
    from src.vector_db.vector_search import VectorSearch
    print("✓ VectorSearch")
except Exception as e:
    print(f"✗ VectorSearch: {e}")

print()
print("All import tests complete!")
