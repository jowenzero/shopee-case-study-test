from typing import Optional, Type
from datetime import datetime, timedelta
import re
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from src.database import DatabaseManager
from src.storage_integration import StorageIntegration


class DateQueryInput(BaseModel):
    date: str = Field(description="Date or date range in YYYY-MM-DD format to query receipts. Can be a single date or range like '2020-01-01 to 2020-12-31'")


class SQLDateQueryTool(BaseTool):
    name: str = "sql_date_query"
    description: str = """
    Query receipts by date or date range.
    Use this when the user asks about purchases on a specific date or date range.
    Input should be a date in YYYY-MM-DD format or a natural language date reference.
    Returns list of receipts with items from that date.
    """
    args_schema: Type[BaseModel] = DateQueryInput
    db: Optional[DatabaseManager] = None

    def __init__(self, db: DatabaseManager):
        super().__init__()
        self.db = db

    def _run(self, date: str) -> str:
        try:
            date_range = self._parse_date_range(date)

            if isinstance(date_range, tuple):
                start_date, end_date = date_range
                receipts = self.db.get_receipts_by_date_range(start_date, end_date)
                date_description = f"{start_date} to {end_date}"
            else:
                receipts = self.db.get_receipts_by_date(date_range)
                date_description = date_range

            if not receipts:
                return f"No receipts found for date {date_description}"

            result = []
            for receipt in receipts:
                items = self.db.get_items_by_receipt(receipt['id'])
                receipt_info = {
                    'receipt_id': receipt['id'],
                    'store_name': receipt['store_name'],
                    'date': receipt['upload_date'],
                    'total_amount': receipt['total_amount'],
                    'items': [{'name': item['item_name'], 'quantity': item['quantity'], 'price': item['price']} for item in items]
                }
                result.append(receipt_info)

            return str(result)
        except Exception as e:
            return f"Error querying by date: {str(e)}"

    async def _arun(self, date: str) -> str:
        raise NotImplementedError("Async not implemented")

    def _parse_date_range(self, date_str: str):
        date_lower = date_str.lower()

        if 'yesterday' in date_lower:
            return (datetime.now().date() - timedelta(days=1)).isoformat()
        elif 'today' in date_lower:
            return datetime.now().date().isoformat()
        elif 'last week' in date_lower or '7 days' in date_lower:
            return (datetime.now().date() - timedelta(days=7)).isoformat()

        range_pattern = r'(\d{4}-\d{2}-\d{2})\s*(?:to|through|until|-)\s*(\d{4}-\d{2}-\d{2})'
        range_match = re.search(range_pattern, date_str)
        if range_match:
            return (range_match.group(1), range_match.group(2))

        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        year_match = re.search(year_pattern, date_str)
        if year_match and any(keyword in date_lower for keyword in ['year', 'entire', 'whole', 'all of']):
            year = year_match.group(1)
            return (f"{year}-01-01", f"{year}-12-31")

        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date().isoformat()
        except:
            return date_str


class ItemQueryInput(BaseModel):
    item_name: str = Field(description="Name of the item to search for")
    days: Optional[int] = Field(default=None, description="Number of days to look back (optional)")


class SQLItemQueryTool(BaseTool):
    name: str = "sql_item_query"
    description: str = """
    Search for receipts containing a specific item.
    Use this when the user asks about a specific food item or product.
    Input should include the item name and optionally the number of days to look back.
    Returns list of receipts containing that item with store names and dates.
    """
    args_schema: Type[BaseModel] = ItemQueryInput
    db: Optional[DatabaseManager] = None

    def __init__(self, db: DatabaseManager):
        super().__init__()
        self.db = db

    def _run(self, item_name: str, days: Optional[int] = None) -> str:
        try:
            items = self.db.search_items_by_name(item_name)

            if days:
                cutoff_date = (datetime.now().date() - timedelta(days=days)).isoformat()
                items = [item for item in items if item['upload_date'] >= cutoff_date]

            if not items:
                return f"No items found matching '{item_name}'"

            result = []
            for item in items:
                item_info = {
                    'item_name': item['item_name'],
                    'store_name': item['store_name'],
                    'date': item['upload_date'],
                    'quantity': item['quantity'],
                    'price': item['price']
                }
                result.append(item_info)

            return str(result)
        except Exception as e:
            return f"Error searching for item: {str(e)}"

    async def _arun(self, item_name: str, days: Optional[int] = None) -> str:
        raise NotImplementedError("Async not implemented")


class ExpenseQueryInput(BaseModel):
    start_date: Optional[str] = Field(default=None, description="Start date in YYYY-MM-DD format (optional)")
    end_date: Optional[str] = Field(default=None, description="End date in YYYY-MM-DD format (optional)")


class SQLExpenseQueryTool(BaseTool):
    name: str = "sql_expense_query"
    description: str = """
    Calculate total expenses for a date range or specific date.
    Use this when the user asks about total spending or expenses.
    Input can include start_date and end_date in YYYY-MM-DD format.
    Returns total expense amount for the specified period.
    """
    args_schema: Type[BaseModel] = ExpenseQueryInput
    db: Optional[DatabaseManager] = None

    def __init__(self, db: DatabaseManager):
        super().__init__()
        self.db = db

    def _run(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        try:
            if start_date and not end_date:
                end_date = datetime.now().date().isoformat()

            total = self.db.get_total_expenses(start_date, end_date)

            if start_date and end_date:
                return f"Total expenses from {start_date} to {end_date}: Rp{total:,.2f}"
            elif start_date:
                return f"Total expenses since {start_date}: Rp{total:,.2f}"
            else:
                return f"Total expenses (all time): Rp{total:,.2f}"
        except Exception as e:
            return f"Error calculating expenses: {str(e)}"

    async def _arun(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        raise NotImplementedError("Async not implemented")


class VectorSearchInput(BaseModel):
    query: str = Field(description="Natural language search query")
    top_k: Optional[int] = Field(default=5, description="Number of results to return")


class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = """
    Semantic search across all receipts and items using natural language.
    Use this when the query is complex or when exact keyword matching might miss relevant results.
    Input should be a natural language query describing what to search for.
    Returns semantically similar receipts and items ranked by relevance.
    """
    args_schema: Type[BaseModel] = VectorSearchInput
    storage: Optional[StorageIntegration] = None

    def __init__(self, storage: StorageIntegration):
        super().__init__()
        self.storage = storage

    def _run(self, query: str, top_k: int = 5) -> str:
        try:
            year_filter = None
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', query)
            if year_match:
                year_filter = year_match.group(1)

            results = self.storage.search_receipts_semantic(query, top_k * 3)

            if year_filter:
                results = [r for r in results if r['upload_date'].startswith(year_filter)]

            if results:
                avg_similarity = sum(r.get('similarity_score', 0) for r in results) / len(results)

                comparative_pattern = r'\b\w+(er|est)\b'
                superlative_pattern = r'\b(most|least)\s+\w+'
                adjective_pattern = r'\b\w+(ive|ous|ful|able|ible)\b'

                has_comparison = bool(re.search(comparative_pattern, query.lower()))
                has_superlative = bool(re.search(superlative_pattern, query.lower()))
                has_adjective = bool(re.search(adjective_pattern, query.lower()))

                if avg_similarity < 0.5 and (has_comparison or has_superlative or has_adjective):
                    negation_prefix = r'\b(in|un|non|im|il|ir)\w+'
                    low_value_suffix = r'\b\w+(less|least)\b'
                    has_negation = bool(re.search(negation_prefix, query.lower()))
                    has_low_indicator = bool(re.search(low_value_suffix, query.lower()))

                    sort_descending = not (has_negation or has_low_indicator)
                    results.sort(key=lambda x: x.get('total_amount', 0), reverse=sort_descending)

            results = results[:top_k]

            if not results:
                return f"No results found for query: '{query}'"

            result = []
            for receipt in results:
                receipt_info = {
                    'receipt_id': receipt['id'],
                    'store_name': receipt['store_name'],
                    'date': receipt['upload_date'],
                    'total_amount': receipt['total_amount'],
                    'similarity_score': receipt.get('similarity_score', 0),
                    'matched_item': receipt.get('matched_item'),
                    'items': [{'name': item['item_name'], 'quantity': item['quantity'], 'price': item['price']} for item in receipt.get('items', [])]
                }
                result.append(receipt_info)

            return str(result)
        except Exception as e:
            return f"Error in semantic search: {str(e)}"

    async def _arun(self, query: str, top_k: int = 5) -> str:
        raise NotImplementedError("Async not implemented")


def create_tools(db: DatabaseManager, storage: StorageIntegration) -> list:
    """
    Create all LangChain tools for receipt querying

    Args:
        db: DatabaseManager instance
        storage: StorageIntegration instance

    Returns:
        List of LangChain tools
    """
    return [
        SQLDateQueryTool(db=db),
        SQLItemQueryTool(db=db),
        SQLExpenseQueryTool(db=db),
        VectorSearchTool(storage=storage)
    ]
