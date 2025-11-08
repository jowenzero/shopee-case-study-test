"""
Cosine Similarity Implementation
No high-level library functions used for similarity calculation.
"""

def dot_product(v1, v2):
    """
    Calculate dot product of two vectors.

    Args:
        v1: First vector (list or array)
        v2: Second vector (list or array)

    Returns:
        float: Dot product of v1 and v2

    Raises:
        ValueError: If vectors have different dimensions
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vectors must have same dimension. Got {len(v1)} and {len(v2)}")

    result = 0.0
    for i in range(len(v1)):
        result += v1[i] * v2[i]

    return result


def magnitude(v):
    """
    Calculate magnitude (L2 norm) of a vector.
    magnitude = sqrt(sum(x^2 for x in vector))

    Args:
        v: Vector (list or array)

    Returns:
        float: Magnitude of the vector
    """
    sum_of_squares = 0.0
    for x in v:
        sum_of_squares += x * x

    return sum_of_squares ** 0.5


def cosine_similarity(v1, v2):
    """
    Calculate cosine similarity between two vectors.
    cosine_sim = dot(v1, v2) / (||v1|| * ||v2||)

    Args:
        v1: First vector (list or array)
        v2: Second vector (list or array)

    Returns:
        float: Cosine similarity [-1, 1]
            1 = identical direction
            0 = orthogonal
            -1 = opposite direction

    Raises:
        ValueError: If vectors have different dimensions or are zero vectors
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vectors must have same dimension. Got {len(v1)} and {len(v2)}")

    dot_prod = dot_product(v1, v2)
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)

    if mag1 == 0 or mag2 == 0:
        raise ValueError("Cannot calculate cosine similarity with zero vector")

    return dot_prod / (mag1 * mag2)


def cosine_distance(v1, v2):
    """
    Calculate cosine distance between two vectors.
    cosine_distance = 1 - cosine_similarity

    Args:
        v1: First vector (list or array)
        v2: Second vector (list or array)

    Returns:
        float: Cosine distance [0, 2]
            0 = identical
            1 = orthogonal
            2 = opposite
    """
    return 1.0 - cosine_similarity(v1, v2)


def batch_cosine_similarity(query_vector, vectors):
    """
    Calculate cosine similarity between a query vector and multiple vectors.

    Args:
        query_vector: Query vector (list or array)
        vectors: List of vectors to compare against

    Returns:
        list: List of similarity scores in same order as input vectors
    """
    similarities = []
    for v in vectors:
        try:
            sim = cosine_similarity(query_vector, v)
            similarities.append(sim)
        except ValueError:
            similarities.append(None)

    return similarities


if __name__ == "__main__":
    # Basic tests
    print("Testing cosine similarity implementation")
    print("=" * 50)

    # Test 1: Identical vectors
    v1 = [1, 2, 3]
    v2 = [1, 2, 3]
    print(f"Identical vectors: {cosine_similarity(v1, v2):.4f} (expected: 1.0000)")

    # Test 2: Orthogonal vectors
    v1 = [1, 0, 0]
    v2 = [0, 1, 0]
    print(f"Orthogonal vectors: {cosine_similarity(v1, v2):.4f} (expected: 0.0000)")

    # Test 3: Opposite vectors
    v1 = [1, 2, 3]
    v2 = [-1, -2, -3]
    print(f"Opposite vectors: {cosine_similarity(v1, v2):.4f} (expected: -1.0000)")

    # Test 4: Similar vectors
    v1 = [1, 2, 3, 4]
    v2 = [1, 2, 3, 5]
    sim = cosine_similarity(v1, v2)
    print(f"Similar vectors: {sim:.4f}")

    # Test 5: Dot product
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    print(f"\nDot product [1,2,3]·[4,5,6]: {dot_product(v1, v2)} (expected: 32)")

    # Test 6: Magnitude
    v = [3, 4]
    print(f"Magnitude [3,4]: {magnitude(v):.4f} (expected: 5.0000)")

    print("\nAll tests completed!")
