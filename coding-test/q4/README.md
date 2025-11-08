# Vector Database with Cosine Similarity

A from-scratch implementation of a vector database with cosine similarity search, built without high-level libraries.

## Overview

This project implements a complete vector database system with the following components:

1. **Cosine Similarity Engine** - Manual implementation using basic math operations
2. **Vector Database** - In-memory storage with CRUD operations and JSON persistence
3. **Search Functionality** - Similarity search with filtering and ranking
4. **Document Similarity Demo** - Text/document search use case demonstration

## Features

- Cosine similarity implementation from scratch
- Full CRUD operations (Create, Read, Update, Delete)
- JSON-based persistence (save/load database to disk)
- Top-k similarity search with metadata filtering
- Near-duplicate detection

## Requirements

- Python >= 3.8
- No external dependencies required

## Installation

```bash
cd q4
# No pip install needed, only use Python standard library
```

## Quick Start

### Run Demo

```bash
python demo.py
```

This runs a complete text/document similarity search demonstration with 20 sample documents across 4 categories.

### Run Tests

```bash
python tests/test_vector_db.py
```

Runs comprehensive test suite covering all components.

## Architecture

### Cosine Similarity Module (`cosine_similarity.py`)

Implements vector similarity from scratch:

```python
# Core functions
dot_product(v1, v2)          # Manual dot product calculation
magnitude(v)                 # Vector magnitude (L2 norm)
cosine_similarity(v1, v2)    # Similarity score [-1, 1]
cosine_distance(v1, v2)      # Distance metric [0, 2]
```

**Formula:**
```
cosine_similarity(A, B) = dot(A, B) / (||A|| * ||B||)
```

### Vector Database (`vector_db.py`)

In-memory vector storage with metadata:

```python
class VectorDB:
    add_vector(id, vector, metadata)      # Add single vector
    add_vectors(dict)                     # Batch add
    get_vector(id)                        # Retrieve by ID
    update_vector(id, vector, metadata)   # Update existing
    delete_vector(id)                     # Remove vector
    filter_by_metadata(filter_fn)         # Filter by metadata
    save(filepath)                        # Persist to JSON
    load(filepath)                        # Load from JSON
    get_stats()                           # Database statistics
```

**Data Structure:**
```json
{
  "name": "document_db",
  "dimension": 128,
  "vectors": {
    "vec_001": {
      "vector": [0.1, 0.2, ...],
      "metadata": {"title": "...", "category": "..."},
      "timestamp": "2025-11-07T..."
    }
  }
}
```

### Vector Search (`vector_search.py`)

Search engine for similarity queries:

```python
class VectorSearch:
    search(query_vector, top_k, filter_fn)    # Main search
    search_by_id(vector_id, top_k)            # Search by existing vector
    batch_search(query_vectors, top_k)        # Multiple queries
    find_duplicates(threshold)                # Find near-duplicates
    get_statistics(query_vector)              # Similarity stats
```

**Search Strategy:**
- Brute-force scan
- Sorts results by similarity score
- Supports metadata filtering

## File Structure

```
q4/
├── src/                     # Source code
│   ├── __init__.py
│   ├── cosine_similarity.py # Core similarity functions
│   ├── vector_db.py         # Database implementation
│   └── vector_search.py     # Search functionality
├── tests/                   # Test suite
│   └── test_vector_db.py
├── demo.py                  # Document similarity demo
└── README.md                # This file
```

## Testing

The test suite (`test_vector_db.py`) includes:

- **Cosine Similarity Tests**: dot product, magnitude, similarity, distance
- **Database Tests**: CRUD operations, persistence, filtering
- **Search Tests**: basic search, filtered search, duplicate detection
- **Edge Cases**: empty database, wrong dimensions, error handling

Run tests:
```bash
python tests/test_vector_db.py
```

Expected output:
```
================================================================================
VECTOR DATABASE TEST SUITE
================================================================================
...
TEST RESULTS: 8/8 tests passed
✓ All tests passed!
```

## Demo

The demo (`demo.py`) showcases text/document similarity search:

- 20 sample documents across 4 categories
- Various search scenarios
- Category filtering
- Duplicate detection
- Statistics display

Run demo:
```bash
python demo.py
```

## Implementation Notes

This project implements cosine similarity from scratch to demonstrate:
1. Understanding of the underlying mathematics
2. No dependency on high-level libraries
3. Educational value for learning vector operations
4. Full control over implementation details
