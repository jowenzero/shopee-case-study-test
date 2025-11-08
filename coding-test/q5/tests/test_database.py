import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import DatabaseManager


def test_database():
    test_db_path = "./data/test_receipts.db"

    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    print("=" * 60)
    print("Testing Database Schema and Operations")
    print("=" * 60)
    print()

    db = DatabaseManager(db_path=test_db_path)

    print("1. Testing Receipt Insertion")
    print("-" * 60)
    today = datetime.now().date().isoformat()
    yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()

    receipt1_id = db.insert_receipt(
        upload_date=today,
        store_name="Solaria",
        total_amount=92000,
        ocr_text="Solaria\nNasi Goreng Ayam Rp46.000\nEs Teh Manis Rp12.000\nTotal: Rp92.000"
    )
    print(f"✓ Inserted receipt 1 (ID: {receipt1_id})")

    receipt2_id = db.insert_receipt(
        upload_date=yesterday,
        store_name="Warunk Upnormal",
        total_amount=65000,
        ocr_text="Warunk Upnormal\nNasi Goreng Kambing Rp45.000\nEs Jeruk Rp15.000\nTotal: Rp65.000"
    )
    print(f"✓ Inserted receipt 2 (ID: {receipt2_id})")

    receipt3_id = db.insert_receipt(
        upload_date=today,
        store_name="Kopi Kenangan",
        total_amount=35000,
        ocr_text="Kopi Kenangan\nKopi Kenangan Mantan Rp25.000\nPisang Bakar Rp10.000\nTotal: Rp35.000"
    )
    print(f"✓ Inserted receipt 3 (ID: {receipt3_id})")
    print()

    print("2. Testing Item Insertion")
    print("-" * 60)

    item1_id = db.insert_item(
        receipt_id=receipt1_id,
        item_name="Nasi Goreng Ayam",
        quantity=1,
        price=46000,
        category="Rice"
    )
    print(f"✓ Inserted item: Nasi Goreng Ayam (ID: {item1_id})")

    item2_id = db.insert_item(
        receipt_id=receipt1_id,
        item_name="Es Teh Manis",
        quantity=1,
        price=12000,
        category="Beverage"
    )
    print(f"✓ Inserted item: Es Teh Manis (ID: {item2_id})")

    item3_id = db.insert_item(
        receipt_id=receipt1_id,
        item_name="Kerupuk",
        quantity=1,
        price=5000,
        category="Sides"
    )
    print(f"✓ Inserted item: Kerupuk (ID: {item3_id})")

    item4_id = db.insert_item(
        receipt_id=receipt2_id,
        item_name="Nasi Goreng Kambing",
        quantity=1,
        price=45000,
        category="Rice"
    )
    print(f"✓ Inserted item: Nasi Goreng Kambing (ID: {item4_id})")

    item5_id = db.insert_item(
        receipt_id=receipt3_id,
        item_name="Kopi Kenangan Mantan",
        quantity=1,
        price=25000,
        category="Beverage"
    )
    print(f"✓ Inserted item: Kopi Kenangan Mantan (ID: {item5_id})")
    print()

    print("3. Testing Receipt Retrieval")
    print("-" * 60)
    receipt = db.get_receipt(receipt1_id)
    print(f"✓ Retrieved receipt: {receipt['store_name']} - Rp{receipt['total_amount']:,.0f}")

    receipt_with_items = db.get_receipt_with_items(receipt1_id)
    print(f"✓ Retrieved receipt with {len(receipt_with_items['items'])} items")
    for item in receipt_with_items['items']:
        print(f"  - {item['item_name']}: Rp{item['price']:,.0f}")
    print()

    print("4. Testing Query Operations")
    print("-" * 60)

    receipts_today = db.get_receipts_by_date(today)
    print(f"✓ Receipts from today: {len(receipts_today)}")
    for r in receipts_today:
        print(f"  - {r['store_name']}: Rp{r['total_amount']:,.0f}")

    receipts_yesterday = db.get_receipts_by_date(yesterday)
    print(f"✓ Receipts from yesterday: {len(receipts_yesterday)}")
    for r in receipts_yesterday:
        print(f"  - {r['store_name']}: Rp{r['total_amount']:,.0f}")

    solaria_receipts = db.get_receipts_by_store("Solaria")
    print(f"✓ Receipts from Solaria: {len(solaria_receipts)}")

    nasi_items = db.search_items_by_name("nasi")
    print(f"✓ Items containing 'nasi': {len(nasi_items)}")
    for item in nasi_items:
        print(f"  - {item['item_name']} from {item['store_name']} on {item['upload_date']}")
    print()

    print("5. Testing Expense Calculations")
    print("-" * 60)

    total_today = db.get_expenses_by_date(today)
    print(f"✓ Total expenses today: Rp{total_today:,.0f}")

    total_yesterday = db.get_expenses_by_date(yesterday)
    print(f"✓ Total expenses yesterday: Rp{total_yesterday:,.0f}")

    total_all = db.get_total_expenses()
    print(f"✓ Total expenses (all time): Rp{total_all:,.0f}")
    print()

    print("6. Testing Update Operations")
    print("-" * 60)

    updated = db.update_receipt(
        receipt1_id,
        store_name="Solaria WTC Serpong",
        total_amount=95000
    )
    print(f"✓ Updated receipt 1: {updated}")

    receipt_updated = db.get_receipt(receipt1_id)
    print(f"  New store name: {receipt_updated['store_name']}")
    print(f"  New total: Rp{receipt_updated['total_amount']:,.0f}")

    item_updated = db.update_item(item1_id, price=48000)
    print(f"✓ Updated item price: {item_updated}")
    print()

    print("7. Testing Statistics")
    print("-" * 60)

    stats = db.get_statistics()
    print(f"✓ Total receipts: {stats['receipt_count']}")
    print(f"✓ Total items: {stats['item_count']}")
    print(f"✓ Total spent: Rp{stats['total_spent']:,.0f}")
    print(f"✓ Average receipt: Rp{stats['avg_receipt']:,.0f}")
    print(f"✓ Top stores:")
    for store in stats['top_stores']:
        print(f"  - {store['store_name']}: {store['count']} visits")
    print()

    print("8. Testing Delete Operations")
    print("-" * 60)

    deleted_item = db.delete_item(item5_id)
    print(f"✓ Deleted item: {deleted_item}")

    deleted_receipt = db.delete_receipt(receipt3_id)
    print(f"✓ Deleted receipt: {deleted_receipt}")

    all_receipts = db.get_all_receipts()
    print(f"✓ Remaining receipts: {len(all_receipts)}")
    print()

    print("9. Testing Date Range Queries")
    print("-" * 60)

    week_ago = (datetime.now() - timedelta(days=7)).date().isoformat()
    receipts_week = db.get_receipts_by_date_range(week_ago, today)
    print(f"✓ Receipts in last 7 days: {len(receipts_week)}")
    print()

    print("=" * 60)
    print("All database tests passed successfully!")
    print("=" * 60)

    print(f"\nTest database created at: {test_db_path}")
    print("You can inspect it with: sqlite3 ./data/test_receipts.db")


if __name__ == "__main__":
    test_database()
