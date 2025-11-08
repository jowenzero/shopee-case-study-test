"""
Test Suite for Vector Database System
Tests cosine similarity, vector database, and search functionality.
"""

import os
import sys
import json

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cosine_similarity import dot_product, magnitude, cosine_similarity, cosine_distance
from src.vector_db import VectorDB
from src.vector_search import VectorSearch


def test_dot_product():
    """Test dot product calculation."""
    print("Testing dot_product...")

    # Test 1: Simple case
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    result = dot_product(v1, v2)
    expected = 32.0  # 1*4 + 2*5 + 3*6
    assert abs(result - expected) < 0.0001, f"Expected {expected}, got {result}"

    # Test 2: Zero vector
    v1 = [0, 0, 0]
    v2 = [1, 2, 3]
    result = dot_product(v1, v2)
    expected = 0.0
    assert abs(result - expected) < 0.0001, f"Expected {expected}, got {result}"

    # Test 3: Negative values
    v1 = [1, -2, 3]
    v2 = [-1, 2, -3]
    result = dot_product(v1, v2)
    expected = -14.0  # 1*-1 + -2*2 + 3*-3
    assert abs(result - expected) < 0.0001, f"Expected {expected}, got {result}"

    # Test 4: Different dimensions should raise error
    try:
        dot_product([1, 2], [1, 2, 3])
        assert False, "Should raise ValueError for different dimensions"
    except ValueError:
        pass

    print("  ✓ All dot_product tests passed")


def test_magnitude():
    """Test magnitude calculation."""
    print("Testing magnitude...")

    # Test 1: Simple case
    v = [3, 4]
    result = magnitude(v)
    expected = 5.0  # sqrt(9 + 16)
    assert abs(result - expected) < 0.0001, f"Expected {expected}, got {result}"

    # Test 2: Unit vector
    v = [1, 0, 0]
    result = magnitude(v)
    expected = 1.0
    assert abs(result - expected) < 0.0001, f"Expected {expected}, got {result}"

    # Test 3: Zero vector
    v = [0, 0, 0]
    result = magnitude(v)
    expected = 0.0
    assert abs(result - expected) < 0.0001, f"Expected {expected}, got {result}"

    # Test 4: Negative values
    v = [-3, -4]
    result = magnitude(v)
    expected = 5.0
    assert abs(result - expected) < 0.0001, f"Expected {expected}, got {result}"

    print("  ✓ All magnitude tests passed")


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    print("Testing cosine_similarity...")

    # Test 1: Identical vectors
    v1 = [1, 2, 3]
    v2 = [1, 2, 3]
    result = cosine_similarity(v1, v2)
    expected = 1.0
    assert abs(result - expected) < 0.0001, f"Expected {expected}, got {result}"

    # Test 2: Orthogonal vectors
    v1 = [1, 0, 0]
    v2 = [0, 1, 0]
    result = cosine_similarity(v1, v2)
    expected = 0.0
    assert abs(result - expected) < 0.0001, f"Expected {expected}, got {result}"

    # Test 3: Opposite vectors
    v1 = [1, 2, 3]
    v2 = [-1, -2, -3]
    result = cosine_similarity(v1, v2)
    expected = -1.0
    assert abs(result - expected) < 0.0001, f"Expected {expected}, got {result}"

    # Test 4: Scaled vectors (should be identical)
    v1 = [1, 2, 3]
    v2 = [2, 4, 6]
    result = cosine_similarity(v1, v2)
    expected = 1.0
    assert abs(result - expected) < 0.0001, f"Expected {expected}, got {result}"

    # Test 5: Zero vector should raise error
    try:
        cosine_similarity([0, 0], [1, 2])
        assert False, "Should raise ValueError for zero vector"
    except ValueError:
        pass

    print("  ✓ All cosine_similarity tests passed")


def test_cosine_distance():
    """Test cosine distance calculation."""
    print("Testing cosine_distance...")

    # Test 1: Identical vectors
    v1 = [1, 2, 3]
    v2 = [1, 2, 3]
    result = cosine_distance(v1, v2)
    expected = 0.0
    assert abs(result - expected) < 0.0001, f"Expected {expected}, got {result}"

    # Test 2: Orthogonal vectors
    v1 = [1, 0, 0]
    v2 = [0, 1, 0]
    result = cosine_distance(v1, v2)
    expected = 1.0
    assert abs(result - expected) < 0.0001, f"Expected {expected}, got {result}"

    # Test 3: Opposite vectors
    v1 = [1, 2, 3]
    v2 = [-1, -2, -3]
    result = cosine_distance(v1, v2)
    expected = 2.0
    assert abs(result - expected) < 0.0001, f"Expected {expected}, got {result}"

    print("  ✓ All cosine_distance tests passed")


def test_vector_db_crud():
    """Test VectorDB CRUD operations."""
    print("Testing VectorDB CRUD operations...")

    # Initialize DB
    db = VectorDB(dimension=3, name="test_db")
    assert len(db) == 0, "New DB should be empty"

    # Test 1: Add vector
    db.add_vector("v1", [1.0, 2.0, 3.0], {"name": "vector1"})
    assert len(db) == 1, "DB should have 1 vector"
    assert "v1" in db, "v1 should be in DB"

    # Test 2: Get vector
    vec = db.get_vector("v1")
    assert vec is not None, "Should retrieve vector"
    assert vec["vector"] == [1.0, 2.0, 3.0], "Vector data should match"
    assert vec["metadata"]["name"] == "vector1", "Metadata should match"

    # Test 3: Add duplicate should raise error
    try:
        db.add_vector("v1", [4.0, 5.0, 6.0])
        assert False, "Should raise ValueError for duplicate ID"
    except ValueError:
        pass

    # Test 4: Add wrong dimension should raise error
    try:
        db.add_vector("v2", [1.0, 2.0], {"name": "wrong_dim"})
        assert False, "Should raise ValueError for wrong dimension"
    except ValueError:
        pass

    # Test 5: Update vector
    db.update_vector("v1", vector=[4.0, 5.0, 6.0])
    vec = db.get_vector("v1")
    assert vec["vector"] == [4.0, 5.0, 6.0], "Vector should be updated"

    # Test 6: Update metadata
    db.update_vector("v1", metadata={"name": "updated"})
    vec = db.get_vector("v1")
    assert vec["metadata"]["name"] == "updated", "Metadata should be updated"

    # Test 7: Delete vector
    result = db.delete_vector("v1")
    assert result == True, "Should return True for successful delete"
    assert len(db) == 0, "DB should be empty after delete"
    assert "v1" not in db, "v1 should not be in DB"

    # Test 8: Batch add
    vectors_dict = {
        "v2": ([1.0, 0.0, 0.0], {"cat": "A"}),
        "v3": ([0.0, 1.0, 0.0], {"cat": "B"}),
        "v4": ([0.0, 0.0, 1.0], {"cat": "A"})
    }
    count = db.add_vectors(vectors_dict)
    assert count == 3, "Should add 3 vectors"
    assert len(db) == 3, "DB should have 3 vectors"

    # Test 9: Filter by metadata
    cat_a = db.filter_by_metadata(lambda m: m.get("cat") == "A")
    assert len(cat_a) == 2, "Should find 2 vectors in category A"

    print("  ✓ All VectorDB CRUD tests passed")


def test_vector_db_persistence():
    """Test VectorDB save/load functionality."""
    print("Testing VectorDB persistence...")

    test_file = "test_save_load.json"

    # Create and populate DB
    db1 = VectorDB(dimension=4, name="save_test")
    db1.add_vector("v1", [1.0, 2.0, 3.0, 4.0], {"type": "test"})
    db1.add_vector("v2", [5.0, 6.0, 7.0, 8.0], {"type": "demo"})

    # Save
    result = db1.save(test_file)
    assert result == True, "Save should succeed"
    assert os.path.exists(test_file), "File should exist"

    # Load into new DB
    db2 = VectorDB(dimension=1, name="empty")  # Wrong dimension, should be overwritten
    result = db2.load(test_file)
    assert result == True, "Load should succeed"

    # Verify data
    assert db2.dimension == 4, "Dimension should match loaded data"
    assert db2.name == "save_test", "Name should match loaded data"
    assert len(db2) == 2, "Should have 2 vectors"
    assert "v1" in db2, "v1 should be loaded"
    assert "v2" in db2, "v2 should be loaded"

    vec = db2.get_vector("v1")
    assert vec["vector"] == [1.0, 2.0, 3.0, 4.0], "Vector data should match"
    assert vec["metadata"]["type"] == "test", "Metadata should match"

    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)

    print("  ✓ All persistence tests passed")


def test_vector_search():
    """Test VectorSearch functionality."""
    print("Testing VectorSearch...")

    # Setup
    db = VectorDB(dimension=3, name="search_test")
    db.add_vector("v1", [1.0, 0.0, 0.0], {"name": "x-axis", "category": "A"})
    db.add_vector("v2", [0.9, 0.1, 0.0], {"name": "near-x", "category": "A"})
    db.add_vector("v3", [0.0, 1.0, 0.0], {"name": "y-axis", "category": "B"})
    db.add_vector("v4", [0.0, 0.0, 1.0], {"name": "z-axis", "category": "A"})

    search = VectorSearch(db)

    # Test 1: Basic search
    results = search.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2, "Should return 2 results"
    assert results[0][0] == "v1", "First result should be v1"
    assert results[1][0] == "v2", "Second result should be v2"
    assert results[0][1] > results[1][1], "Results should be sorted by similarity"

    # Test 2: Search by ID
    results = search.search_by_id("v1", top_k=2, exclude_self=True)
    assert len(results) == 2, "Should return 2 results"
    assert "v1" not in [r[0] for r in results], "Should exclude query vector"

    # Test 3: Filtered search
    results = search.search(
        [1.0, 0.0, 0.0],
        top_k=10,
        filter_fn=lambda m: m.get("category") == "A"
    )
    assert len(results) == 3, "Should only return category A vectors"
    for vid, sim, meta in results:
        assert meta["category"] == "A", "All results should be category A"

    # Test 4: Find duplicates
    db.add_vector("v5", [0.95, 0.05, 0.0], {"name": "almost-x", "category": "A"})
    duplicates = search.find_duplicates(threshold=0.98)
    assert len(duplicates) > 0, "Should find near-duplicates"

    # Test 5: Statistics
    stats = search.get_statistics([1.0, 0.0, 0.0])
    assert stats["count"] == 5, "Should compute stats for all vectors"
    assert "mean" in stats, "Should include mean"
    assert "min" in stats, "Should include min"
    assert "max" in stats, "Should include max"
    assert stats["max"] == 1.0, "Max similarity should be 1.0"

    print("  ✓ All VectorSearch tests passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")

    # Test 1: Empty database search
    db = VectorDB(dimension=2, name="empty_db")
    search = VectorSearch(db)
    results = search.search([1.0, 0.0], top_k=5)
    assert len(results) == 0, "Should return empty results for empty DB"

    # Test 2: Top-k larger than database
    db.add_vector("v1", [1.0, 0.0], {})
    results = search.search([1.0, 0.0], top_k=100)
    assert len(results) == 1, "Should return all available results"

    # Test 3: Wrong dimension in search
    try:
        search.search([1.0, 2.0, 3.0], top_k=5)
        assert False, "Should raise ValueError for wrong dimension"
    except ValueError:
        pass

    # Test 4: Search for non-existent ID
    try:
        search.search_by_id("nonexistent", top_k=5)
        assert False, "Should raise ValueError for non-existent ID"
    except ValueError:
        pass

    print("  ✓ All edge case tests passed")


def run_all_tests():
    """Run all test suites."""
    print("=" * 80)
    print("VECTOR DATABASE TEST SUITE")
    print("=" * 80)

    tests = [
        ("Cosine Similarity", [
            test_dot_product,
            test_magnitude,
            test_cosine_similarity,
            test_cosine_distance
        ]),
        ("Vector Database", [
            test_vector_db_crud,
            test_vector_db_persistence
        ]),
        ("Vector Search", [
            test_vector_search,
            test_edge_cases
        ])
    ]

    total_tests = 0
    passed_tests = 0

    for suite_name, suite_tests in tests:
        print(f"\n{suite_name} Tests")
        print("-" * 80)

        for test_func in suite_tests:
            total_tests += 1
            try:
                test_func()
                passed_tests += 1
            except AssertionError as e:
                print(f"  ✗ {test_func.__name__} FAILED: {e}")
            except Exception as e:
                print(f"  ✗ {test_func.__name__} ERROR: {e}")

    print()
    print("=" * 80)
    print(f"TEST RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 80)

    if passed_tests == total_tests:
        print("✓ All tests passed!")
        return True
    else:
        print(f"✗ {total_tests - passed_tests} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
