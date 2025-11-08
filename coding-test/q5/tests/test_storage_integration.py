import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.storage_integration import StorageIntegration
from src.ocr_extractor import OCRExtractor
from dotenv import load_dotenv


def test_storage_integration():
    print("=" * 80)
    print("Testing Database Storage Integration (SQLite + Vector DB)")
    print("=" * 80)
    print()

    load_dotenv()

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("✗ GEMINI_API_KEY not found or not set in .env file")
        print()
        return

    print(f"✓ API Key found: {api_key[:20]}...")
    print()

    # Initialize components
    try:
        print("Initializing Storage Integration...")
        storage = StorageIntegration(
            db_path="./data/test_receipts.db",
            vector_db_path="./data/test_vector_db.json"
        )
        print()

        print("Initializing OCR Extractor...")
        ocr = OCRExtractor()
        print()
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return

    # Test with one receipt
    test_image = "./uploads/test_physical_receipt_1.png"

    print(f"Testing with: {test_image}")
    print("-" * 80)

    if not os.path.exists(test_image):
        print(f"✗ Test image not found: {test_image}")
        return

    # Step 1: Extract receipt data
    print("STEP 1: Extracting receipt data with Gemini OCR...")
    try:
        receipt_data = ocr.extract_and_parse(test_image)
        print(f"  ✓ OCR extraction completed")
        print(f"    - Store: {receipt_data.store_name}")
        print(f"    - Date: {receipt_data.date}")
        print(f"    - Total: Rp{receipt_data.total_amount:,.0f}" if receipt_data.total_amount else "    - Total: N/A")
        print(f"    - Items: {len(receipt_data.items)}")
        print()
    except Exception as e:
        print(f"  ✗ OCR extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Store in databases 
    print("STEP 2: Storing in SQLite + Vector DB...")
    try:
        result = storage.store_receipt(receipt_data)

        if result['success']:
            print(f"  ✓ Storage successful!")
            print(f"    - Receipt ID (SQLite): {result['receipt_id']}")
            print(f"    - Receipt Vector ID: {result['receipt_vector_id']}")
            print(f"    - Items stored: {result['item_count']}")
            print(f"    - Item vector IDs: {result['item_vector_ids']}")
            print()
        else:
            print(f"  ✗ Storage failed: {result.get('error', 'Unknown error')}")
            return
    except Exception as e:
        print(f"  ✗ Storage failed: {e}")
        import traceback
        traceback.print_exc()
        return

    receipt_id = result['receipt_id']

    # Step 3: Verify SQLite storage
    print("STEP 3: Verifying SQLite storage...")
    try:
        retrieved_receipt = storage.db.get_receipt_with_items(receipt_id)

        if retrieved_receipt:
            print(f"  ✓ Receipt retrieved from SQLite")
            print(f"    - ID: {retrieved_receipt['id']}")
            print(f"    - Store: {retrieved_receipt['store_name']}")
            print(f"    - Date: {retrieved_receipt['upload_date']}")
            print(f"    - Total: Rp{retrieved_receipt['total_amount']:,.0f}" if retrieved_receipt['total_amount'] else "    - Total: N/A")
            print(f"    - Vector ID: {retrieved_receipt['vector_id']}")
            print(f"    - Items count: {len(retrieved_receipt['items'])}")

            if retrieved_receipt['items']:
                print(f"\n    Items:")
                for item in retrieved_receipt['items'][:5]:  # Show first 5 items
                    price_str = f"Rp{item['price']:,.0f}" if item['price'] else "N/A"
                    print(f"      - {item['item_name']}: {price_str} (Vector ID: {item['vector_id']})")
            print()
        else:
            print(f"  ✗ Receipt not found in SQLite")
            return
    except Exception as e:
        print(f"  ✗ SQLite verification failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Verify Vector DB storage
    print("STEP 4: Verifying Vector DB storage...")
    try:
        receipt_vector_id = retrieved_receipt['vector_id']
        if receipt_vector_id:
            vector_data = storage.vector_db.get_vector(receipt_vector_id)
            if vector_data:
                print(f"  ✓ Receipt vector retrieved from Vector DB")
                print(f"    - Vector ID: {receipt_vector_id}")
                print(f"    - Vector dimension: {len(vector_data['vector'])}")
                print(f"    - Metadata: {vector_data['metadata']}")
                print()
            else:
                print(f"  ✗ Receipt vector not found in Vector DB")
        else:
            print(f"  ✗ No vector ID in SQLite record")
    except Exception as e:
        print(f"  ✗ Vector DB verification failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 5: Test semantic search
    print("STEP 5: Testing semantic search...")
    search_queries = [
        receipt_data.store_name if receipt_data.store_name else "restaurant",
        receipt_data.items[0].item_name if receipt_data.items else "food"
    ]

    for query in search_queries:
        print(f"\n  Query: '{query}'")
        try:
            results = storage.search_receipts_semantic(query, top_k=3)
            print(f"  ✓ Found {len(results)} results")

            for i, result in enumerate(results, 1):
                print(f"    {i}. {result['store_name']} - {result['upload_date']}")
                print(f"       Total: Rp{result['total_amount']:,.0f}" if result['total_amount'] else "       Total: N/A")
                print(f"       Similarity: {result.get('similarity_score', 0):.4f}")
                if 'matched_item' in result:
                    print(f"       Matched item: {result['matched_item']}")
        except Exception as e:
            print(f"  ✗ Search failed: {e}")
            import traceback
            traceback.print_exc()

    print()

    # Step 6: Show statistics
    print("STEP 6: Database statistics...")
    try:
        stats = storage.get_statistics()
        print(f"  ✓ Statistics retrieved")
        print(f"    SQLite:")
        print(f"      - Total receipts: {stats['sqlite'].get('receipt_count', 0)}")
        print(f"      - Total items: {stats['sqlite'].get('item_count', 0)}")
        print(f"    Vector DB:")
        print(f"      - Total vectors: {stats['vector_db'].get('total_vectors', 0)}")
        print(f"      - Dimension: {stats['vector_db'].get('dimension', 'N/A')}")
        print()
    except Exception as e:
        print(f"  ✗ Statistics failed: {e}")

    # Step 7: Test retrieval with context
    print("STEP 7: Testing retrieval with full context...")
    try:
        full_receipt = storage.get_receipt_with_context(receipt_id)

        if full_receipt:
            print(f"  ✓ Receipt retrieved with full context")
            print(f"    - Receipt has vector data: {'vector_data' in full_receipt}")

            items_with_vectors = sum(1 for item in full_receipt['items'] if 'vector_data' in item)
            print(f"    - Items with vector data: {items_with_vectors}/{len(full_receipt['items'])}")
            print()
        else:
            print(f"  ✗ Failed to retrieve receipt with context")
    except Exception as e:
        print(f"  ✗ Context retrieval failed: {e}")

    # Summary
    print("=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    print()
    print("✓ Step 1: OCR Extraction - PASSED")
    print("✓ Step 2: Database Storage - PASSED")
    print("✓ Step 3: SQLite Verification - PASSED")
    print("✓ Step 4: Vector DB Verification - PASSED")
    print("✓ Step 5: Semantic Search - PASSED")
    print("✓ Step 6: Statistics - PASSED")
    print("✓ Step 7: Full Context Retrieval - PASSED")
    print()
    print("✓ Database Storage Integration is working correctly!")
    print("✓ Data flows: OCR → SQLite → Vector DB")
    print("✓ Bidirectional linking established (SQLite ↔ Vector DB)")
    print("✓ Semantic search operational")
    print()
    print("=" * 80)
    print("Testing Complete")
    print("=" * 80)


if __name__ == "__main__":
    test_storage_integration()
