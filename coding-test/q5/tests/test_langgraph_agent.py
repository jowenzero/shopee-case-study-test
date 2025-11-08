import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import DatabaseManager
from src.storage_integration import StorageIntegration
from src.langgraph_agent import ReceiptQueryAgent


def test_agent_workflow():
    """Test the complete LangGraph agent workflow"""
    print("\n" + "="*60)
    print("Testing LangGraph Agent Workflow")
    print("="*60)

    db = DatabaseManager(db_path="./data/receipts.db")
    storage = StorageIntegration(
        db_path="./data/receipts.db",
        vector_db_path="./data/vector_db.json"
    )

    print("\nInitializing ReceiptQueryAgent...")
    agent = ReceiptQueryAgent(db=db, storage=storage)
    print("✓ Agent initialized successfully")

    test_queries = [
        "What food did I buy on 2023-04-15?",
        "Give me total expenses for 2023",
        "Where did I buy chicken from?",
        "Show me expensive purchases"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test Query {i}: {query}")
        print('='*60)

        try:
            result = agent.query(query)

            print(f"\n[Intent]: {result['intent']}")
            print(f"\n[Tool Results]:\n{result['tool_results'][:300]}..." if len(result['tool_results']) > 300 else f"\n[Tool Results]:\n{result['tool_results']}")
            print(f"\n[Response]:\n{result['response']}")

            if i < len(test_queries):
                print(f"\nWaiting 20 seconds to respect rate limit...")
                time.sleep(20)

        except Exception as e:
            print(f"\n✗ Error processing query: {e}")
            import traceback
            traceback.print_exc()

            if i < len(test_queries):
                print(f"\nWaiting 20 seconds before next query...")
                time.sleep(20)


def test_router_classification():
    """Test the router node intent classification"""
    print("\n" + "="*60)
    print("Testing Router Node Intent Classification")
    print("="*60)

    db = DatabaseManager(db_path="./data/receipts.db")
    storage = StorageIntegration(
        db_path="./data/receipts.db",
        vector_db_path="./data/vector_db.json"
    )

    agent = ReceiptQueryAgent(db=db, storage=storage)

    test_cases = [
        ("What did I buy yesterday?", "date_query"),
        ("Show me receipts from 2023-04-15", "date_query"),
        ("Find me nasi goreng", "item_query"),
        ("Where did I buy chicken?", "item_query"),
        ("Total expenses for April 2023", "expense_query"),
        ("How much did I spend?", "expense_query"),
        ("Most expensive food purchases", "semantic_query")
    ]

    for idx, (query, expected_intent) in enumerate(test_cases, 1):
        state = {"query": query, "intent": "", "tool_results": "", "response": "", "messages": []}
        state = agent._router_node(state)

        intent = state["intent"]
        match = "✓" if expected_intent in intent else "✗"

        print(f"\n{match} Query: '{query}'")
        print(f"  Expected: {expected_intent}, Got: {intent}")

        if idx % 3 == 0 and idx < len(test_cases):
            print(f"\nWaiting 8 seconds to respect rate limit...")
            time.sleep(8)


if __name__ == "__main__":
    print("LangGraph Agent Testing")
    print("="*60)

    db = DatabaseManager(db_path="./data/receipts.db")
    stats = db.get_statistics()

    if stats['receipt_count'] == 0:
        print("\nWARNING: No receipts in database. Please run test_storage_integration.py first.")
        exit(1)

    print(f"\nDatabase contains {stats['receipt_count']} receipts and {stats['item_count']} items")

    try:
        test_router_classification()

        print("\n" + "="*60)
        print("Waiting 20 seconds before workflow tests to respect rate limit...")
        print("="*60)
        time.sleep(20)

        test_agent_workflow()

        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
