"""
Vector Search Implementation
Search for similar vectors using cosine similarity.
"""

from typing import List, Tuple, Optional, Callable
from .vector_db import VectorDB
from .cosine_similarity import cosine_similarity


class VectorSearch:
    """
    Search engine for finding similar vectors in a VectorDB.
    """

    def __init__(self, db: VectorDB):
        """
        Initialize search engine.

        Args:
            db: VectorDB instance to search
        """
        self.db = db

    def search(self,
               query_vector: List[float],
               top_k: int = 5,
               filter_fn: Optional[Callable] = None) -> List[Tuple[str, float, dict]]:
        """
        Search for most similar vectors to query.

        Args:
            query_vector: Query vector to search for
            top_k: Number of results to return
            filter_fn: Optional function to filter vectors by metadata

        Returns:
            list: List of (vector_id, similarity_score, metadata) tuples,
                  sorted by similarity (highest first)

        Raises:
            ValueError: If query vector dimension doesn't match DB
        """
        if len(query_vector) != self.db.dimension:
            raise ValueError(f"Query dimension {len(query_vector)} doesn't match DB dimension {self.db.dimension}")

        # Get candidate vectors (optionally filtered)
        if filter_fn:
            candidate_ids = self.db.filter_by_metadata(filter_fn)
        else:
            candidate_ids = self.db.get_all_ids()

        if not candidate_ids:
            return []

        # Calculate similarities
        results = []
        for vid in candidate_ids:
            entry = self.db.get_vector(vid)
            vector = entry["vector"]
            metadata = entry["metadata"]

            try:
                similarity = cosine_similarity(query_vector, vector)
                results.append((vid, similarity, metadata))
            except ValueError:
                # Skip if similarity calculation fails
                continue

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        return results[:top_k]

    def search_by_id(self,
                     vector_id: str,
                     top_k: int = 5,
                     exclude_self: bool = True,
                     filter_fn: Optional[Callable] = None) -> List[Tuple[str, float, dict]]:
        """
        Search for vectors similar to a vector already in the DB.

        Args:
            vector_id: ID of vector to use as query
            top_k: Number of results to return
            exclude_self: If True, exclude the query vector from results
            filter_fn: Optional function to filter vectors by metadata

        Returns:
            list: List of (vector_id, similarity_score, metadata) tuples

        Raises:
            ValueError: If vector_id not found
        """
        query_vector = self.db.get_vector_data(vector_id)
        if query_vector is None:
            raise ValueError(f"Vector ID '{vector_id}' not found")

        results = self.search(query_vector, top_k + (1 if exclude_self else 0), filter_fn)

        # Remove the query vector itself if requested
        if exclude_self:
            results = [(vid, sim, meta) for vid, sim, meta in results if vid != vector_id]

        return results[:top_k]

    def batch_search(self,
                     query_vectors: List[Tuple[str, List[float]]],
                     top_k: int = 5) -> dict:
        """
        Search for multiple query vectors.

        Args:
            query_vectors: List of (query_id, vector) tuples
            top_k: Number of results per query

        Returns:
            dict: Mapping query_id to list of results
        """
        results = {}
        for query_id, query_vector in query_vectors:
            results[query_id] = self.search(query_vector, top_k)

        return results

    def find_duplicates(self, threshold: float = 0.99) -> List[Tuple[str, str, float]]:
        """
        Find near-duplicate vectors in the database.

        Args:
            threshold: Similarity threshold for considering vectors as duplicates

        Returns:
            list: List of (id1, id2, similarity) tuples for near-duplicates
        """
        duplicates = []
        all_ids = self.db.get_all_ids()

        for i, id1 in enumerate(all_ids):
            v1 = self.db.get_vector_data(id1)

            for id2 in all_ids[i+1:]:
                v2 = self.db.get_vector_data(id2)

                try:
                    sim = cosine_similarity(v1, v2)
                    if sim >= threshold:
                        duplicates.append((id1, id2, sim))
                except ValueError:
                    continue

        return duplicates

    def get_statistics(self, query_vector: List[float]) -> dict:
        """
        Get similarity statistics for a query vector.

        Args:
            query_vector: Query vector

        Returns:
            dict: Statistics including mean, min, max similarity
        """
        if len(query_vector) != self.db.dimension:
            raise ValueError(f"Query dimension {len(query_vector)} doesn't match DB dimension {self.db.dimension}")

        similarities = []
        for vid in self.db.get_all_ids():
            vector = self.db.get_vector_data(vid)
            try:
                sim = cosine_similarity(query_vector, vector)
                similarities.append(sim)
            except ValueError:
                continue

        if not similarities:
            return {}

        return {
            "count": len(similarities),
            "mean": sum(similarities) / len(similarities),
            "min": min(similarities),
            "max": max(similarities)
        }


if __name__ == "__main__":
    # Basic tests
    print("Testing VectorSearch")
    print("=" * 50)

    # Create test database
    db = VectorDB(dimension=4, name="test_search")

    # Add test vectors
    db.add_vector("doc1", [1.0, 0.0, 0.0, 0.0], {"title": "Document 1", "category": "tech"})
    db.add_vector("doc2", [0.9, 0.1, 0.0, 0.0], {"title": "Document 2", "category": "tech"})
    db.add_vector("doc3", [0.0, 1.0, 0.0, 0.0], {"title": "Document 3", "category": "science"})
    db.add_vector("doc4", [0.0, 0.0, 1.0, 0.0], {"title": "Document 4", "category": "tech"})
    db.add_vector("doc5", [0.8, 0.2, 0.0, 0.0], {"title": "Document 5", "category": "tech"})

    print(f"Created database with {len(db)} vectors\n")

    # Initialize search
    search = VectorSearch(db)

    # Test 1: Basic search
    query = [1.0, 0.0, 0.0, 0.0]
    results = search.search(query, top_k=3)

    print("Test 1: Search for similar to [1.0, 0.0, 0.0, 0.0]")
    for vid, sim, meta in results:
        print(f"  {vid}: {sim:.4f} - {meta['title']}")

    # Test 2: Search by ID
    print("\nTest 2: Find similar to doc1")
    results = search.search_by_id("doc1", top_k=3)
    for vid, sim, meta in results:
        print(f"  {vid}: {sim:.4f} - {meta['title']}")

    # Test 3: Filtered search
    print("\nTest 3: Search with category filter (tech only)")
    results = search.search(query, top_k=3, filter_fn=lambda m: m.get("category") == "tech")
    for vid, sim, meta in results:
        print(f"  {vid}: {sim:.4f} - {meta['title']} ({meta['category']})")

    # Test 4: Find duplicates
    print("\nTest 4: Find near-duplicates (threshold=0.95)")
    duplicates = search.find_duplicates(threshold=0.95)
    for id1, id2, sim in duplicates:
        print(f"  {id1} <-> {id2}: {sim:.4f}")

    # Test 5: Statistics
    print("\nTest 5: Similarity statistics")
    stats = search.get_statistics(query)
    print(f"  Count: {stats['count']}")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")

    print("\nAll tests completed!")
