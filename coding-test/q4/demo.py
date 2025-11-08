"""
Text/Document Similarity Demo
Demonstrates vector database usage for document similarity search.
"""

import os
import sys
import random

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vector_db import VectorDB
from src.vector_search import VectorSearch


def generate_document_embedding(doc_id: int, category: str, dimension: int = 128) -> list:
    """
    Generate a simulated document embedding vector.

    In a real application, this would use:
    - Sentence transformers (e.g., BERT, RoBERTa)
    - OpenAI embeddings
    - Word2Vec/GloVe

    For demo purposes, we generate vectors with category-specific patterns.

    Args:
        doc_id: Document identifier
        category: Document category (affects vector pattern)
        dimension: Embedding dimension

    Returns:
        list: Simulated embedding vector
    """
    random.seed(doc_id)

    # Create base vector
    vector = [random.gauss(0, 0.1) for _ in range(dimension)]

    # Add category-specific patterns
    if category == "technology":
        # Emphasize first quarter of dimensions
        for i in range(dimension // 4):
            vector[i] += random.uniform(0.5, 1.0)
    elif category == "science":
        # Emphasize second quarter
        for i in range(dimension // 4, dimension // 2):
            vector[i] += random.uniform(0.5, 1.0)
    elif category == "business":
        # Emphasize third quarter
        for i in range(dimension // 2, 3 * dimension // 4):
            vector[i] += random.uniform(0.5, 1.0)
    elif category == "sports":
        # Emphasize fourth quarter
        for i in range(3 * dimension // 4, dimension):
            vector[i] += random.uniform(0.5, 1.0)

    return vector


def create_sample_documents():
    """
    Create sample document collection.

    Returns:
        list: List of (doc_id, title, content_snippet, category) tuples
    """
    documents = [
        # Technology
        ("tech_001", "Introduction to Machine Learning",
         "Machine learning is a subset of artificial intelligence...", "technology"),
        ("tech_002", "Deep Learning Neural Networks",
         "Neural networks are computing systems inspired by biological...", "technology"),
        ("tech_003", "Cloud Computing Fundamentals",
         "Cloud computing delivers computing services over the internet...", "technology"),
        ("tech_004", "Quantum Computing Basics",
         "Quantum computers use quantum mechanical phenomena...", "technology"),
        ("tech_005", "Artificial Intelligence in Healthcare",
         "AI is transforming healthcare through predictive analytics...", "technology"),

        # Science
        ("sci_001", "Climate Change Research",
         "Global climate patterns are shifting due to greenhouse gases...", "science"),
        ("sci_002", "Genetic Engineering Advances",
         "CRISPR technology enables precise gene editing...", "science"),
        ("sci_003", "Space Exploration Missions",
         "Recent Mars missions have discovered evidence of water...", "science"),
        ("sci_004", "Renewable Energy Sources",
         "Solar and wind power are becoming more efficient...", "science"),
        ("sci_005", "Neuroscience Discoveries",
         "Brain research reveals new insights into consciousness...", "science"),

        # Business
        ("bus_001", "Startup Funding Strategies",
         "Venture capital and angel investors provide early stage funding...", "business"),
        ("bus_002", "E-commerce Growth Trends",
         "Online retail continues to expand globally...", "business"),
        ("bus_003", "Digital Marketing Strategies",
         "Social media and content marketing drive customer engagement...", "business"),
        ("bus_004", "Supply Chain Management",
         "Logistics optimization improves delivery efficiency...", "business"),
        ("bus_005", "Financial Technology Innovation",
         "Fintech companies are disrupting traditional banking...", "business"),

        # Sports
        ("spo_001", "Olympic Games History",
         "The Olympics showcase athletic excellence worldwide...", "sports"),
        ("spo_002", "Professional Basketball Analysis",
         "NBA statistics reveal player performance trends...", "sports"),
        ("spo_003", "Soccer World Cup Preview",
         "National teams prepare for the upcoming tournament...", "sports"),
        ("spo_004", "Marathon Training Techniques",
         "Endurance running requires proper training methods...", "sports"),
        ("spo_005", "Sports Nutrition Guidelines",
         "Athletes optimize performance through diet planning...", "sports"),
    ]

    return documents


def main():
    """Run the document similarity demo."""

    print("=" * 80)
    print("TEXT/DOCUMENT SIMILARITY SEARCH DEMO")
    print("=" * 80)
    print()

    # Configuration
    DIMENSION = 128
    TOP_K = 5

    print(f"Configuration:")
    print(f"  Vector dimension: {DIMENSION}")
    print(f"  Database: In-memory with JSON persistence")
    print(f"  Similarity metric: Cosine similarity (implemented from scratch)")
    print()

    # Step 1: Create database
    print("Step 1: Creating vector database...")
    db = VectorDB(dimension=DIMENSION, name="document_db")

    # Step 2: Add documents
    print("Step 2: Adding documents to database...")
    documents = create_sample_documents()

    for doc_id, title, snippet, category in documents:
        # Generate simulated embedding
        embedding = generate_document_embedding(hash(doc_id), category, DIMENSION)

        # Add to database
        metadata = {
            "title": title,
            "snippet": snippet,
            "category": category
        }
        db.add_vector(doc_id, embedding, metadata)

    print(f"  Added {len(db)} documents")
    print()

    # Step 3: Display database stats
    print("Step 3: Database statistics")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Step 4: Save database
    print("Step 4: Saving database to disk...")
    db.save("document_db.json")
    print("  Saved to document_db.json")
    print()

    # Step 5: Initialize search
    print("Step 5: Initializing search engine...")
    search = VectorSearch(db)
    print("  Search engine ready")
    print()

    # Step 6: Demo searches
    print("=" * 80)
    print("SEARCH DEMONSTRATIONS")
    print("=" * 80)
    print()

    # Demo 1: Search similar to a specific document
    print("Demo 1: Find documents similar to 'Introduction to Machine Learning'")
    print("-" * 80)
    results = search.search_by_id("tech_001", top_k=TOP_K, exclude_self=True)

    for i, (doc_id, similarity, meta) in enumerate(results, 1):
        print(f"{i}. [{doc_id}] {meta['title']}")
        print(f"   Similarity: {similarity:.4f} | Category: {meta['category']}")
        print(f"   {meta['snippet'][:60]}...")
        print()

    # Demo 2: Category-filtered search
    print("Demo 2: Find similar technology documents")
    print("-" * 80)
    query_doc = db.get_vector_data("tech_002")
    results = search.search(
        query_doc,
        top_k=TOP_K,
        filter_fn=lambda m: m.get("category") == "technology"
    )

    for i, (doc_id, similarity, meta) in enumerate(results, 1):
        print(f"{i}. [{doc_id}] {meta['title']}")
        print(f"   Similarity: {similarity:.4f}")
        print()

    # Demo 3: Find documents in different categories
    print("Demo 3: Cross-category comparison - Science document vs all")
    print("-" * 80)
    query_doc_id = "sci_001"
    results = search.search_by_id(query_doc_id, top_k=TOP_K, exclude_self=True)

    query_meta = db.get_vector("query_doc_id")
    print(f"Query: {db.get_vector(query_doc_id)['metadata']['title']}")
    print()

    for i, (doc_id, similarity, meta) in enumerate(results, 1):
        print(f"{i}. [{doc_id}] {meta['title']}")
        print(f"   Similarity: {similarity:.4f} | Category: {meta['category']}")
        print()

    # Demo 4: Find near-duplicates
    print("Demo 4: Find near-duplicate documents (similarity > 0.95)")
    print("-" * 80)
    duplicates = search.find_duplicates(threshold=0.95)

    if duplicates:
        for id1, id2, sim in duplicates[:5]:  # Show max 5
            doc1 = db.get_vector(id1)
            doc2 = db.get_vector(id2)
            print(f"  {doc1['metadata']['title']}")
            print(f"  <-> {doc2['metadata']['title']}")
            print(f"  Similarity: {sim:.4f}")
            print()
    else:
        print("  No near-duplicates found")
        print()

    # Demo 5: Statistics
    print("Demo 5: Similarity statistics for a technology document")
    print("-" * 80)
    query_vec = db.get_vector_data("tech_001")
    stats = search.get_statistics(query_vec)

    print(f"  Total comparisons: {stats['count']}")
    print(f"  Average similarity: {stats['mean']:.4f}")
    print(f"  Min similarity: {stats['min']:.4f}")
    print(f"  Max similarity: {stats['max']:.4f}")
    print()

    # Demo 6: Category distribution
    print("Demo 6: Documents by category")
    print("-" * 80)
    categories = {}
    for doc_id in db.get_all_ids():
        category = db.get_vector(doc_id)['metadata']['category']
        categories[category] = categories.get(category, 0) + 1

    for category, count in sorted(categories.items()):
        print(f"  {category.capitalize()}: {count} documents")
    print()

    print("=" * 80)
    print("DEMO COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
