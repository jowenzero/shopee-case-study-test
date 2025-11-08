"""
Vector Database Implementation
In-memory vector storage with CRUD operations and JSON persistence.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any


class VectorDB:
    """
    Simple in-memory vector database.

    Stores vectors with metadata and supports CRUD operations.
    """

    def __init__(self, dimension: int, name: str = "vector_db"):
        """
        Initialize vector database.

        Args:
            dimension: Dimension of vectors to store
            name: Name of the database
        """
        self.dimension = dimension
        self.name = name
        self.vectors = {}
        self.created_at = datetime.now().isoformat()

    def add_vector(self, vector_id: str, vector: List[float], metadata: Optional[Dict] = None) -> bool:
        """
        Add a single vector to the database.

        Args:
            vector_id: Unique identifier for the vector
            vector: Vector data (list of floats)
            metadata: Optional metadata dictionary

        Returns:
            bool: True if added successfully

        Raises:
            ValueError: If vector dimension doesn't match or ID already exists
        """
        if vector_id in self.vectors:
            raise ValueError(f"Vector ID '{vector_id}' already exists")

        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} doesn't match DB dimension {self.dimension}")

        self.vectors[vector_id] = {
            "vector": list(vector),
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }

        return True

    def add_vectors(self, vectors_dict: Dict[str, tuple]) -> int:
        """
        Batch add multiple vectors.

        Args:
            vectors_dict: Dictionary mapping IDs to (vector, metadata) tuples

        Returns:
            int: Number of vectors successfully added
        """
        count = 0
        for vector_id, (vector, metadata) in vectors_dict.items():
            try:
                self.add_vector(vector_id, vector, metadata)
                count += 1
            except ValueError as e:
                print(f"Warning: Skipping {vector_id}: {e}")

        return count

    def get_vector(self, vector_id: str) -> Optional[Dict]:
        """
        Retrieve a vector by ID.

        Args:
            vector_id: Vector identifier

        Returns:
            dict: Vector data with metadata, or None if not found
        """
        return self.vectors.get(vector_id)

    def get_vector_data(self, vector_id: str) -> Optional[List[float]]:
        """
        Get just the vector data (without metadata).

        Args:
            vector_id: Vector identifier

        Returns:
            list: Vector data, or None if not found
        """
        entry = self.vectors.get(vector_id)
        return entry["vector"] if entry else None

    def update_vector(self, vector_id: str, vector: Optional[List[float]] = None,
                     metadata: Optional[Dict] = None) -> bool:
        """
        Update an existing vector and/or its metadata.

        Args:
            vector_id: Vector identifier
            vector: New vector data (optional)
            metadata: New metadata (optional)

        Returns:
            bool: True if updated successfully

        Raises:
            ValueError: If vector ID doesn't exist or dimension mismatch
        """
        if vector_id not in self.vectors:
            raise ValueError(f"Vector ID '{vector_id}' not found")

        if vector is not None:
            if len(vector) != self.dimension:
                raise ValueError(f"Vector dimension {len(vector)} doesn't match DB dimension {self.dimension}")
            self.vectors[vector_id]["vector"] = list(vector)

        if metadata is not None:
            self.vectors[vector_id]["metadata"] = metadata

        self.vectors[vector_id]["timestamp"] = datetime.now().isoformat()

        return True

    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector from the database.

        Args:
            vector_id: Vector identifier

        Returns:
            bool: True if deleted, False if not found
        """
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            return True
        return False

    def get_all_ids(self) -> List[str]:
        """Get list of all vector IDs."""
        return list(self.vectors.keys())

    def get_all_vectors(self) -> Dict[str, List[float]]:
        """Get all vectors without metadata."""
        return {vid: v["vector"] for vid, v in self.vectors.items()}

    def filter_by_metadata(self, filter_fn) -> List[str]:
        """
        Filter vectors by metadata.

        Args:
            filter_fn: Function that takes metadata dict and returns bool

        Returns:
            list: Vector IDs matching the filter
        """
        matching_ids = []
        for vid, data in self.vectors.items():
            if filter_fn(data["metadata"]):
                matching_ids.append(vid)

        return matching_ids

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            dict: Statistics including count, dimension, memory usage estimate
        """
        total_vectors = len(self.vectors)
        memory_estimate_mb = (total_vectors * self.dimension * 8) / (1024 * 1024)  # 8 bytes per float

        return {
            "name": self.name,
            "dimension": self.dimension,
            "total_vectors": total_vectors,
            "created_at": self.created_at,
            "memory_estimate_mb": round(memory_estimate_mb, 2)
        }

    def save(self, filepath: str) -> bool:
        """
        Save database to JSON file.

        Args:
            filepath: Path to save file

        Returns:
            bool: True if saved successfully
        """
        data = {
            "name": self.name,
            "dimension": self.dimension,
            "created_at": self.created_at,
            "vectors": self.vectors
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False

    def load(self, filepath: str) -> bool:
        """
        Load database from JSON file.

        Args:
            filepath: Path to load file

        Returns:
            bool: True if loaded successfully
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.name = data["name"]
            self.dimension = data["dimension"]
            self.created_at = data["created_at"]
            self.vectors = data["vectors"]

            return True
        except Exception as e:
            print(f"Error loading database: {e}")
            return False

    def __len__(self):
        """Return number of vectors in database."""
        return len(self.vectors)

    def __contains__(self, vector_id):
        """Check if vector ID exists in database."""
        return vector_id in self.vectors

    def __repr__(self):
        return f"VectorDB(name='{self.name}', dimension={self.dimension}, vectors={len(self.vectors)})"


if __name__ == "__main__":
    # Basic tests
    print("Testing VectorDB")
    print("=" * 50)

    # Initialize DB
    db = VectorDB(dimension=3, name="test_db")
    print(f"Created: {db}")

    # Add vectors
    db.add_vector("v1", [1.0, 2.0, 3.0], {"type": "test", "category": "A"})
    db.add_vector("v2", [4.0, 5.0, 6.0], {"type": "test", "category": "B"})
    db.add_vector("v3", [7.0, 8.0, 9.0], {"type": "demo", "category": "A"})
    print(f"Added 3 vectors: {db}")

    # Get vector
    vec = db.get_vector("v1")
    print(f"\nRetrieved v1: {vec['vector']}, metadata: {vec['metadata']}")

    # Update vector
    db.update_vector("v1", metadata={"type": "updated", "category": "C"})
    print(f"Updated v1 metadata: {db.get_vector('v1')['metadata']}")

    # Filter by metadata
    category_a = db.filter_by_metadata(lambda m: m.get("category") == "A")
    print(f"\nVectors in category A: {category_a}")

    # Stats
    stats = db.get_stats()
    print(f"\nDatabase stats: {stats}")

    # Save and load
    db.save("test_db.json")
    print("\nSaved to test_db.json")

    new_db = VectorDB(dimension=3, name="loaded_db")
    new_db.load("test_db.json")
    print(f"Loaded: {new_db}")
    print(f"Loaded vectors: {new_db.get_all_ids()}")

    # Delete vector
    db.delete_vector("v2")
    print(f"\nAfter deleting v2: {db}")

    print("\nAll tests completed!")
