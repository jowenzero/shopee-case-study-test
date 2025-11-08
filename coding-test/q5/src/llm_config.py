import os
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_gemini_llm(
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None
) -> ChatGoogleGenerativeAI:
    """
    Initialize Gemini LLM for LangChain

    Args:
        model_name: Gemini model name
        temperature: Temperature for response generation (0.0 = deterministic)
        max_tokens: Maximum tokens in response

    Returns:
        ChatGoogleGenerativeAI instance
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        raise ValueError("GEMINI_API_KEY not found or invalid in .env file")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        google_api_key=api_key
    )

    return llm
