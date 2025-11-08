import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager


class DatabaseManager:
    def __init__(self, db_path: str = "./data/receipts.db"):
        self.db_path = db_path
        self._ensure_db_directory()
        self.init_database()

    def _ensure_db_directory(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def init_database(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS receipts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    upload_date TEXT NOT NULL,
                    store_name TEXT,
                    total_amount REAL,
                    ocr_text TEXT,
                    created_at TEXT NOT NULL,
                    vector_id TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    receipt_id INTEGER NOT NULL,
                    item_name TEXT NOT NULL,
                    quantity REAL DEFAULT 1.0,
                    price REAL,
                    category TEXT,
                    created_at TEXT NOT NULL,
                    vector_id TEXT,
                    FOREIGN KEY (receipt_id) REFERENCES receipts (id) ON DELETE CASCADE
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_receipts_upload_date
                ON receipts(upload_date)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_receipts_store_name
                ON receipts(store_name)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_items_receipt_id
                ON items(receipt_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_items_item_name
                ON items(item_name)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_items_category
                ON items(category)
            """)

    def insert_receipt(
        self,
        upload_date: str,
        store_name: Optional[str] = None,
        total_amount: Optional[float] = None,
        ocr_text: Optional[str] = None,
        vector_id: Optional[str] = None
    ) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            created_at = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO receipts
                (upload_date, store_name, total_amount, ocr_text, created_at, vector_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (upload_date, store_name, total_amount, ocr_text, created_at, vector_id))

            return cursor.lastrowid

    def insert_item(
        self,
        receipt_id: int,
        item_name: str,
        quantity: float = 1.0,
        price: Optional[float] = None,
        category: Optional[str] = None,
        vector_id: Optional[str] = None
    ) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            created_at = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO items
                (receipt_id, item_name, quantity, price, category, created_at, vector_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (receipt_id, item_name, quantity, price, category, created_at, vector_id))

            return cursor.lastrowid

    def get_receipt(self, receipt_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM receipts WHERE id = ?", (receipt_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_receipt_with_items(self, receipt_id: int) -> Optional[Dict[str, Any]]:
        receipt = self.get_receipt(receipt_id)
        if not receipt:
            return None

        items = self.get_items_by_receipt(receipt_id)
        receipt['items'] = items
        return receipt

    def get_items_by_receipt(self, receipt_id: int) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM items WHERE receipt_id = ?", (receipt_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_all_receipts(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM receipts ORDER BY upload_date DESC"
            if limit:
                query += f" LIMIT {limit} OFFSET {offset}"
            cursor.execute(query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_receipts_by_date(self, date: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM receipts WHERE upload_date = ? ORDER BY created_at DESC",
                (date,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_receipts_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM receipts WHERE upload_date BETWEEN ? AND ? ORDER BY upload_date DESC",
                (start_date, end_date)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_receipts_by_store(self, store_name: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM receipts WHERE store_name LIKE ? ORDER BY upload_date DESC",
                (f"%{store_name}%",)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def search_items_by_name(self, item_name: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT items.*, receipts.store_name, receipts.upload_date
                FROM items
                JOIN receipts ON items.receipt_id = receipts.id
                WHERE items.item_name LIKE ?
                ORDER BY receipts.upload_date DESC
                """,
                (f"%{item_name}%",)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_total_expenses(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> float:
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if start_date and end_date:
                cursor.execute(
                    "SELECT SUM(total_amount) as total FROM receipts WHERE upload_date BETWEEN ? AND ?",
                    (start_date, end_date)
                )
            elif start_date:
                cursor.execute(
                    "SELECT SUM(total_amount) as total FROM receipts WHERE upload_date >= ?",
                    (start_date,)
                )
            else:
                cursor.execute("SELECT SUM(total_amount) as total FROM receipts")

            result = cursor.fetchone()
            return result['total'] if result['total'] else 0.0

    def get_expenses_by_date(self, date: str) -> float:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT SUM(total_amount) as total FROM receipts WHERE upload_date = ?",
                (date,)
            )
            result = cursor.fetchone()
            return result['total'] if result['total'] else 0.0

    def update_receipt(
        self,
        receipt_id: int,
        store_name: Optional[str] = None,
        total_amount: Optional[float] = None,
        ocr_text: Optional[str] = None,
        vector_id: Optional[str] = None
    ) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()

            updates = []
            params = []

            if store_name is not None:
                updates.append("store_name = ?")
                params.append(store_name)
            if total_amount is not None:
                updates.append("total_amount = ?")
                params.append(total_amount)
            if ocr_text is not None:
                updates.append("ocr_text = ?")
                params.append(ocr_text)
            if vector_id is not None:
                updates.append("vector_id = ?")
                params.append(vector_id)

            if not updates:
                return False

            params.append(receipt_id)
            query = f"UPDATE receipts SET {', '.join(updates)} WHERE id = ?"

            cursor.execute(query, params)
            return cursor.rowcount > 0

    def update_item(
        self,
        item_id: int,
        item_name: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        category: Optional[str] = None,
        vector_id: Optional[str] = None
    ) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()

            updates = []
            params = []

            if item_name is not None:
                updates.append("item_name = ?")
                params.append(item_name)
            if quantity is not None:
                updates.append("quantity = ?")
                params.append(quantity)
            if price is not None:
                updates.append("price = ?")
                params.append(price)
            if category is not None:
                updates.append("category = ?")
                params.append(category)
            if vector_id is not None:
                updates.append("vector_id = ?")
                params.append(vector_id)

            if not updates:
                return False

            params.append(item_id)
            query = f"UPDATE items SET {', '.join(updates)} WHERE id = ?"

            cursor.execute(query, params)
            return cursor.rowcount > 0

    def delete_receipt(self, receipt_id: int) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))
            return cursor.rowcount > 0

    def delete_item(self, item_id: int) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM items WHERE id = ?", (item_id,))
            return cursor.rowcount > 0

    def get_statistics(self) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as count FROM receipts")
            receipt_count = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) as count FROM items")
            item_count = cursor.fetchone()['count']

            cursor.execute("SELECT SUM(total_amount) as total FROM receipts")
            total_spent = cursor.fetchone()['total'] or 0.0

            cursor.execute("SELECT AVG(total_amount) as avg FROM receipts")
            avg_receipt = cursor.fetchone()['avg'] or 0.0

            cursor.execute("""
                SELECT store_name, COUNT(*) as count
                FROM receipts
                WHERE store_name IS NOT NULL
                GROUP BY store_name
                ORDER BY count DESC
                LIMIT 5
            """)
            top_stores = [dict(row) for row in cursor.fetchall()]

            return {
                'receipt_count': receipt_count,
                'item_count': item_count,
                'total_spent': round(total_spent, 2),
                'avg_receipt': round(avg_receipt, 2),
                'top_stores': top_stores
            }

    def clear_all_data(self) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM items")
            cursor.execute("DELETE FROM receipts")
            return True
