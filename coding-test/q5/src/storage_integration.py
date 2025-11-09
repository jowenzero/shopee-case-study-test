from typing import Dict, Any, Optional
import os
from datetime import datetime

from src.database import DatabaseManager
from src.embeddings import EmbeddingGenerator
from src.vector_db.vector_db import VectorDB
from src.vector_db.vector_search import VectorSearch
from src.ocr_extractor import ReceiptData


class StorageIntegration:
    """
    Integrates OCR extraction with SQLite and Vector DB storage
    """

    def __init__(
        self,
        db_path: str = "./data/receipts.db",
        vector_db_path: str = "./data/vector_db.json"
    ):
        """
        Initialize storage integration

        Args:
            db_path: Path to SQLite database
            vector_db_path: Path to Vector database file
        """
        # Initialize SQLite database
        self.db = DatabaseManager(db_path=db_path)
        print(f"✓ SQLite database initialized: {db_path}")

        # Initialize embedding generator first to get dimension
        self.embedding_gen = EmbeddingGenerator()

        # Get embedding dimension from the model
        sample_embedding = self.embedding_gen.generate_query_embedding("test")
        embedding_dimension = len(sample_embedding)

        # Initialize Vector database
        self.vector_db_path = vector_db_path
        self.vector_db = VectorDB(dimension=embedding_dimension, name="receipts_vector_db")

        # Load existing vector database if it exists
        if os.path.exists(vector_db_path):
            self.vector_db.load(vector_db_path)
            print(f"✓ Vector database loaded: {vector_db_path}")
        else:
            print(f"✓ Vector database initialized: {vector_db_path}")

        # Initialize vector search engine
        self.vector_search = VectorSearch(self.vector_db)

    def store_receipt(
        self,
        receipt_data: ReceiptData
    ) -> Dict[str, Any]:
        """
        Store receipt data in both SQLite and Vector DB

        Args:
            receipt_data: Extracted receipt data from OCR

        Returns:
            Dictionary with storage results including receipt_id and vector_ids
        """
        try:
            # Prepare receipt data for SQLite
            upload_date = receipt_data.date or datetime.now().date().isoformat()

            # Insert into SQLite receipts table (no image_path)
            receipt_id = self.db.insert_receipt(
                upload_date=upload_date,
                store_name=receipt_data.store_name,
                total_amount=receipt_data.total_amount,
                ocr_text=receipt_data.raw_text
            )

            print(f"  ✓ Receipt inserted into SQLite (ID: {receipt_id})")

            # Generate receipt embedding
            receipt_dict = {
                'store_name': receipt_data.store_name,
                'date': receipt_data.date,
                'total_amount': receipt_data.total_amount,
                'items': [item.to_dict() for item in receipt_data.items]
            }

            receipt_embedding = self.embedding_gen.generate_receipt_embedding(receipt_dict)

            # Store receipt embedding in Vector DB
            receipt_metadata = {
                'type': 'receipt',
                'receipt_id': receipt_id,
                'store_name': receipt_data.store_name,
                'date': receipt_data.date,
                'total_amount': receipt_data.total_amount
            }

            # Generate unique vector ID
            receipt_vector_id = f"receipt_{receipt_id}"

            self.vector_db.add_vector(
                vector_id=receipt_vector_id,
                vector=receipt_embedding.tolist(),
                metadata=receipt_metadata
            )

            print(f"  ✓ Receipt embedding stored in Vector DB (ID: {receipt_vector_id})")

            # Update SQLite receipt with vector_id
            self.db.update_receipt(receipt_id, vector_id=receipt_vector_id)

            # Store items
            item_vector_ids = []
            for item in receipt_data.items:
                # Insert item into SQLite
                item_id = self.db.insert_item(
                    receipt_id=receipt_id,
                    item_name=item.item_name,
                    quantity=item.quantity,
                    price=item.price,
                    category=item.category
                )

                # Generate item embedding
                item_embedding = self.embedding_gen.generate_item_embedding(item.to_dict())

                # Store item embedding in Vector DB
                item_metadata = {
                    'type': 'item',
                    'item_id': item_id,
                    'receipt_id': receipt_id,
                    'item_name': item.item_name,
                    'store_name': receipt_data.store_name,
                    'date': receipt_data.date
                }

                # Generate unique vector ID for item
                item_vector_id = f"item_{item_id}"

                self.vector_db.add_vector(
                    vector_id=item_vector_id,
                    vector=item_embedding.tolist(),
                    metadata=item_metadata
                )

                # Update SQLite item with vector_id
                self.db.update_item(item_id, vector_id=item_vector_id)

                item_vector_ids.append(item_vector_id)

            print(f"  ✓ {len(receipt_data.items)} items stored in both databases")

            # Save vector database to disk
            self.vector_db.save(self.vector_db_path)

            return {
                'success': True,
                'receipt_id': receipt_id,
                'receipt_vector_id': receipt_vector_id,
                'item_count': len(receipt_data.items),
                'item_vector_ids': item_vector_ids
            }

        except Exception as e:
            print(f"  ✗ Storage error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_receipt_with_context(self, receipt_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve receipt with all related data from both databases

        Args:
            receipt_id: Receipt ID from SQLite

        Returns:
            Dictionary with receipt data, items, and vector context
        """
        # Get receipt with items from SQLite
        receipt = self.db.get_receipt_with_items(receipt_id)

        if not receipt:
            return None

        # Get receipt vector from Vector DB if available
        if receipt.get('vector_id'):
            receipt_vector = self.vector_db.get_vector(receipt['vector_id'])
            receipt['vector_data'] = receipt_vector

        # Get item vectors
        for item in receipt['items']:
            if item.get('vector_id'):
                item_vector = self.vector_db.get_vector(item['vector_id'])
                item['vector_data'] = item_vector

        return receipt

    def search_receipts_semantic(self, query: str, top_k: int = 5) -> list:
        """
        Semantic search for receipts using Vector DB

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of receipts matching the query
        """
        # Generate query embedding
        query_embedding = self.embedding_gen.generate_query_embedding(query)

        # Search in Vector DB using VectorSearch
        # Returns list of (vector_id, similarity_score, metadata) tuples
        results = self.vector_search.search(
            query_vector=query_embedding.tolist(),
            top_k=top_k * 2  # Get more results to account for duplicates
        )

        # Enrich results with SQLite data
        enriched_results = []
        seen_receipt_ids = set()

        for vector_id, similarity_score, metadata in results:
            if metadata.get('type') == 'receipt':
                receipt_id = metadata.get('receipt_id')
                if receipt_id and receipt_id not in seen_receipt_ids:
                    receipt = self.db.get_receipt_with_items(receipt_id)
                    if receipt:
                        receipt['similarity_score'] = similarity_score
                        enriched_results.append(receipt)
                        seen_receipt_ids.add(receipt_id)

            elif metadata.get('type') == 'item':
                receipt_id = metadata.get('receipt_id')
                if receipt_id and receipt_id not in seen_receipt_ids:
                    receipt = self.db.get_receipt_with_items(receipt_id)
                    if receipt:
                        receipt['similarity_score'] = similarity_score
                        receipt['matched_item'] = metadata.get('item_name')
                        enriched_results.append(receipt)
                        seen_receipt_ids.add(receipt_id)

            # Stop if we have enough unique receipts
            if len(enriched_results) >= top_k:
                break

        return enriched_results[:top_k]

    def delete_receipt(self, receipt_id: int) -> Dict[str, Any]:
        """
        Delete receipt from both SQLite and Vector DB

        Args:
            receipt_id: Receipt ID to delete

        Returns:
            Dictionary with deletion result
        """
        try:
            receipt = self.db.get_receipt_with_items(receipt_id)
            if not receipt:
                return {
                    'success': False,
                    'error': 'Receipt not found'
                }

            if receipt.get('vector_id'):
                self.vector_db.delete_vector(receipt['vector_id'])
                print(f"  ✓ Deleted receipt vector: {receipt['vector_id']}")

            for item in receipt.get('items', []):
                if item.get('vector_id'):
                    self.vector_db.delete_vector(item['vector_id'])

            item_count = len(receipt.get('items', []))
            if item_count > 0:
                print(f"  ✓ Deleted {item_count} item vectors")

            success = self.db.delete_receipt(receipt_id)

            if success:
                self.vector_db.save(self.vector_db_path)
                print(f"  ✓ Receipt {receipt_id} deleted from both databases")
                return {
                    'success': True,
                    'receipt_id': receipt_id,
                    'items_deleted': item_count
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to delete from SQLite'
                }

        except Exception as e:
            print(f"  ✗ Delete error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics from both databases

        Returns:
            Dictionary with statistics
        """
        sqlite_stats = self.db.get_statistics()
        vector_stats = self.vector_db.get_stats()

        return {
            'sqlite': sqlite_stats,
            'vector_db': vector_stats,
            'total_receipts': sqlite_stats.get('receipt_count', 0),
            'total_items': sqlite_stats.get('item_count', 0),
            'total_vectors': vector_stats.get('total_vectors', 0)
        }
