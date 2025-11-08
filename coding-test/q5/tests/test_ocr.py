import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ocr_extractor import OCRExtractor
from dotenv import load_dotenv


def test_gemini_ocr():
    print("=" * 80)
    print("Testing Gemini Vision API OCR on All Test Receipts")
    print("=" * 80)
    print()

    load_dotenv()

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("✗ GEMINI_API_KEY not found or not set in .env file")
        print()
        print("Please create a .env file with your Gemini API key:")
        print("  GEMINI_API_KEY=your_actual_api_key_here")
        print()
        return

    print(f"✓ API Key found: {api_key[:20]}...")
    print()

    # Initialize Gemini OCR
    try:
        extractor = OCRExtractor()
        print()
    except Exception as e:
        print(f"✗ Failed to initialize Gemini OCR: {e}")
        return

    test_receipts = [
        {
            "name": "Online Receipt 1",
            "file": "./uploads/test_online_receipt_1.jpg",
            "type": "Online/Digital"
        },
        {
            "name": "Online Receipt 2",
            "file": "./uploads/test_online_receipt_2.jpg",
            "type": "Online/Digital"
        },
        {
            "name": "Physical Receipt 1",
            "file": "./uploads/test_physical_receipt_1.png",
            "type": "Physical/Scanned"
        },
        {
            "name": "Physical Receipt 2",
            "file": "./uploads/test_physical_receipt_2.png",
            "type": "Physical/Scanned"
        }
    ]

    results = []

    for idx, receipt_info in enumerate(test_receipts, 1):
        print(f"[{idx}/4] Processing: {receipt_info['name']}")
        print(f"      Type: {receipt_info['type']}")
        print(f"      File: {receipt_info['file']}")
        print("-" * 80)

        if not os.path.exists(receipt_info['file']):
            print(f"  ✗ File not found!")
            print()
            results.append({
                "name": receipt_info['name'],
                "success": False,
                "error": "File not found"
            })
            continue

        try:
            print(f"  → Calling Gemini Vision API...")
            receipt_data = extractor.extract_and_parse(receipt_info['file'])

            # Display results
            print(f"  ✓ Extraction completed!")
            print()

            if receipt_data.store_name:
                print(f"  ✓ Store: {receipt_data.store_name}")
            else:
                print(f"  ✗ Store: Not detected")

            if receipt_data.date:
                print(f"  ✓ Date: {receipt_data.date}")
            else:
                print(f"  ✗ Date: Not detected")

            if receipt_data.total_amount:
                print(f"  ✓ Total: Rp{receipt_data.total_amount:,.0f}")
            else:
                print(f"  ✗ Total: Not detected")

            if receipt_data.subtotal:
                print(f"  ✓ Subtotal: Rp{receipt_data.subtotal:,.0f}")

            if receipt_data.tax:
                print(f"  ✓ Tax/Service: Rp{receipt_data.tax:,.0f}")

            print(f"  ✓ Items Detected: {len(receipt_data.items)}")

            if len(receipt_data.items) > 0:
                print(f"\n  Items:")
                for item in receipt_data.items:
                    price_str = f"Rp{item.price:,.0f}" if item.price else "N/A"
                    qty_str = f"{item.quantity}x" if item.quantity != 1.0 else "1x"
                    print(f"    - {qty_str} {item.item_name}: {price_str}")
            else:
                print(f"  ⚠ No items detected")

            # Quality assessment
            summary = extractor.get_extraction_summary(receipt_data)
            quality_indicator = "✓" if summary['quality'] == 'good' else "⚠"
            print(f"\n  {quality_indicator} Extraction Quality: {summary['quality']}")
            print(f"    - Has store name: {summary['has_store_name']}")
            print(f"    - Has date: {summary['has_date']}")
            print(f"    - Has total: {summary['has_total']}")
            print(f"    - Items with prices: {summary['items_with_prices']}/{summary['item_count']}")

            # Show database-ready format
            print(f"\n  Database Format Preview:")
            print(f"    receipts table -> store: {receipt_data.store_name}, date: {receipt_data.date}, total: {receipt_data.total_amount}")
            print(f"    items table -> {len(receipt_data.items)} items ready for insertion")

            results.append({
                "name": receipt_info['name'],
                "success": True,
                "confidence": receipt_data.confidence,
                "store": receipt_data.store_name,
                "date": receipt_data.date,
                "total": receipt_data.total_amount,
                "subtotal": receipt_data.subtotal,
                "tax": receipt_data.tax,
                "items_count": len(receipt_data.items),
                "quality": summary['quality']
            })

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": receipt_info['name'],
                "success": False,
                "error": str(e)
            })

        print()

    # Summary Report
    print("=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    print()

    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    print(f"Total Receipts Tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()

    if successful:
        print("Successful Extractions:")
        print("-" * 80)
        for result in successful:
            print(f"  {result['name']}:")
            print(f"    Store: {result.get('store', 'N/A')}")
            print(f"    Date: {result.get('date', 'N/A')}")
            print(f"    Total: Rp{result.get('total', 0):,.0f}" if result.get('total') else "    Total: N/A")
            if result.get('subtotal'):
                print(f"    Subtotal: Rp{result.get('subtotal', 0):,.0f}")
            if result.get('tax'):
                print(f"    Tax: Rp{result.get('tax', 0):,.0f}")
            print(f"    Items: {result.get('items_count', 0)}")
            print(f"    Quality: {result.get('quality', 'unknown')}")
            print()

    if failed:
        print("Failed Extractions:")
        print("-" * 80)
        for result in failed:
            print(f"  {result['name']}: {result.get('error', 'Unknown error')}")
        print()

    # Overall assessment
    success_rate = (len(successful) / len(results)) * 100 if results else 0
    print(f"Success Rate: {success_rate:.0f}%")

    if success_rate >= 75:
        print("✓ Gemini Vision OCR is working excellently!")
        print("✓ Data is ready for database storage (Step 4)")
    elif success_rate >= 50:
        print("⚠ Gemini Vision OCR needs improvement")
    else:
        print("✗ Gemini Vision OCR has significant issues")

    print()
    print("=" * 80)
    print("Testing Complete")
    print("=" * 80)


if __name__ == "__main__":
    test_gemini_ocr()
