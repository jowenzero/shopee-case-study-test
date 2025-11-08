import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, timedelta
from src.database import DatabaseManager
from src.storage_integration import StorageIntegration
from src.langchain_tools import (
    SQLDateQueryTool,
    SQLItemQueryTool,
    SQLExpenseQueryTool,
    VectorSearchTool,
    create_tools
)


def setup_test_data():
    """
    Setup test database with sample data
    """
    db = DatabaseManager(db_path="./data/receipts.db")

    print("Setting up test data...")
    print("Database should have existing receipts from previous tests")

    stats = db.get_statistics()
    print(f"Found {stats['receipt_count']} receipts in database")
    print(f"Found {stats['item_count']} items in database")

    return db


def test_sql_date_query_tool():
    print("\n" + "="*60)
    print("TEST 1: SQLDateQueryTool")
    print("="*60)

    db = DatabaseManager(db_path="./data/receipts.db")
    tool = SQLDateQueryTool(db=db)

    print("\nTool Name:", tool.name)
    print("Tool Description:", tool.description)

    test_dates = [
        "2020-01-01",
        "2023-04-15",
        "2025-01-01",
        "yesterday",
        "today",
        "2020-01-01 to 2020-12-31",
        "2023-04-01 to 2023-04-30",
        "entire year 2023"
    ]

    for test_date in test_dates:
        print(f"\n--- Testing with date: {test_date} ---")
        result = tool._run(test_date)
        print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")


def test_sql_item_query_tool():
    print("\n" + "="*60)
    print("TEST 2: SQLItemQueryTool")
    print("="*60)

    db = DatabaseManager(db_path="./data/receipts.db")
    tool = SQLItemQueryTool(db=db)

    print("\nTool Name:", tool.name)
    print("Tool Description:", tool.description)

    test_items = [
        ("nasi", None),
        ("ayam", 7),
        ("goreng", None)
    ]

    for item_name, days in test_items:
        print(f"\n--- Testing with item: '{item_name}', days: {days} ---")
        result = tool._run(item_name, days)
        print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")


def test_sql_expense_query_tool():
    print("\n" + "="*60)
    print("TEST 3: SQLExpenseQueryTool")
    print("="*60)

    db = DatabaseManager(db_path="./data/receipts.db")
    tool = SQLExpenseQueryTool(db=db)

    print("\nTool Name:", tool.name)
    print("Tool Description:", tool.description)

    test_queries = [
        (None, None),
        ("2020-01-01", "2020-12-31"),
        ("2023-01-01", "2023-12-31"),
        ("2025-01-01", "2025-12-31"),
        ("2023-04-01", "2023-04-30")
    ]

    for start_date, end_date in test_queries:
        print(f"\n--- Testing with start_date: {start_date}, end_date: {end_date} ---")
        result = tool._run(start_date, end_date)
        print(f"Result: {result}")


def test_vector_search_tool():
    print("\n" + "="*60)
    print("TEST 4: VectorSearchTool")
    print("="*60)

    storage = StorageIntegration(
        db_path="./data/receipts.db",
        vector_db_path="./data/vector_db.json"
    )
    tool = VectorSearchTool(storage=storage)

    print("\nTool Name:", tool.name)
    print("Tool Description:", tool.description)

    test_queries = [
        "fried rice",
        "chicken",
        "most expensive purchases",
        "food from 2023"
    ]

    for query in test_queries:
        print(f"\n--- Testing semantic search: '{query}' ---")
        result = tool._run(query, top_k=3)
        print(f"Result: {result[:300]}..." if len(result) > 300 else f"Result: {result}")


def test_create_tools():
    print("\n" + "="*60)
    print("TEST 5: create_tools Function")
    print("="*60)

    db = DatabaseManager(db_path="./data/receipts.db")
    storage = StorageIntegration(
        db_path="./data/receipts.db",
        vector_db_path="./data/vector_db.json"
    )

    tools = create_tools(db, storage)

    print(f"\nCreated {len(tools)} tools:")
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool.name}: {tool.description[:100]}...")


if __name__ == "__main__":
    print("Testing LangChain Tools")
    print("="*60)

    db = setup_test_data()

    if db.get_statistics()['receipt_count'] == 0:
        print("\nWARNING: No receipts in database. Please run test_storage_integration.py first to populate test data.")
        exit(1)

    try:
        test_sql_date_query_tool()
        test_sql_item_query_tool()
        test_sql_expense_query_tool()
        test_vector_search_tool()
        test_create_tools()

        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
