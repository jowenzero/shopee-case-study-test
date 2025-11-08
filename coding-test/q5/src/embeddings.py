from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np


class EmbeddingGenerator:
    """Generate embeddings for receipt text using Sentence Transformers"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding model

        Args:
            model_name: Name of the sentence-transformers model
                       'all-MiniLM-L6-v2' is a good balance of speed and quality
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"✓ Embedding model loaded successfully")

    def generate_receipt_embedding(self, receipt_data: Dict[str, Any]) -> np.ndarray:
        """
        Generate embedding for entire receipt

        Creates a text summary of the receipt and generates embedding

        Args:
            receipt_data: Dictionary containing store_name, date, total_amount, items

        Returns:
            Embedding vector as numpy array
        """
        # Create text summary of receipt
        text_parts = []

        if receipt_data.get('store_name'):
            text_parts.append(f"Store: {receipt_data['store_name']}")

        if receipt_data.get('date'):
            text_parts.append(f"Date: {receipt_data['date']}")

        if receipt_data.get('total_amount'):
            text_parts.append(f"Total: Rp{receipt_data['total_amount']:,.0f}")

        # Add item names
        if receipt_data.get('items'):
            item_names = [item.get('item_name', '') for item in receipt_data['items'] if item.get('item_name')]
            if item_names:
                text_parts.append(f"Items: {', '.join(item_names)}")

        # Combine into single text
        receipt_text = ". ".join(text_parts)

        # Generate embedding
        embedding = self.model.encode(receipt_text, convert_to_numpy=True)

        return embedding

    def generate_item_embedding(self, item_data: Dict[str, Any]) -> np.ndarray:
        """
        Generate embedding for individual item

        Args:
            item_data: Dictionary containing item_name, quantity, price

        Returns:
            Embedding vector as numpy array
        """
        # Create text representation of item
        text_parts = []

        item_name = item_data.get('item_name', '')
        if item_name:
            text_parts.append(item_name)

        quantity = item_data.get('quantity', 1.0)
        if quantity and quantity != 1.0:
            text_parts.append(f"quantity: {quantity}")

        price = item_data.get('price')
        if price:
            text_parts.append(f"price: Rp{price:,.0f}")

        category = item_data.get('category')
        if category:
            text_parts.append(f"category: {category}")

        # Combine into single text
        item_text = ", ".join(text_parts)

        # Generate embedding
        embedding = self.model.encode(item_text, convert_to_numpy=True)

        return embedding

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for search query

        Args:
            query: Natural language query string

        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding

    def generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently

        Args:
            texts: List of text strings

        Returns:
            Array of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
