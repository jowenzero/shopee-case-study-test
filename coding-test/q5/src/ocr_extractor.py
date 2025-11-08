import google.generativeai as genai
from PIL import Image
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ReceiptItem:
    item_name: str
    quantity: float = 1.0
    price: Optional[float] = None
    category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReceiptData:
    raw_text: str
    store_name: Optional[str] = None
    date: Optional[str] = None
    total_amount: Optional[float] = None
    items: List[ReceiptItem] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    confidence: float = 0.0

    def __post_init__(self):
        if self.items is None:
            self.items = []

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['items'] = [item.to_dict() for item in self.items]
        return data


class OCRExtractor:
    """OCR Extractor using Google Gemini Vision API"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini API"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in .env file or pass as parameter."
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Use Gemini 2.5 Flash for vision
        self.model = genai.GenerativeModel('gemini-2.5-flash')

        print("✓ Gemini OCR initialized successfully")

    def extract_and_parse(self, image_path: str) -> ReceiptData:
        """Extract and parse receipt data from image using Gemini Vision API"""

        # Load image
        image = Image.open(image_path)

        # Create structured prompt for Gemini
        prompt = """You are an expert at extracting structured data from food receipt images.

Analyze this receipt image and extract the following information in JSON format:

{
  "store_name": "Name of the store/restaurant",
  "date": "Date in YYYY-MM-DD format",
  "total_amount": Total amount as a number (without currency symbol or dots/commas),
  "subtotal": Subtotal amount as a number (if available, otherwise null),
  "tax": Tax/service charge amount as a number (if available, otherwise null),
  "items": [
    {
      "item_name": "Name of the item",
      "quantity": Quantity as a number,
      "price": Price as a number (without currency symbol or dots/commas)
    }
  ]
}

Important rules:
1. For Indonesian Rupiah amounts like "Rp88.602", convert to plain number: 88602 (remove "Rp" and dots)
2. For amounts like "173.734", convert to: 173734
3. If quantity is not explicitly shown, use 1.0
4. Extract ALL items from the receipt, not just a few
5. If a field cannot be determined, use null
6. For dates, convert any format to YYYY-MM-DD (e.g., "15 Apr 2023" becomes "2023-04-15")
7. Store name should be the main business name, not location or branch details
8. Do not include summary lines like "Subtotal", "Total", "Tax" as items

Return ONLY the JSON object, no additional text."""

        try:
            # Call Gemini Vision API
            response = self.model.generate_content([prompt, image])

            # Extract JSON from response
            response_text = response.text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            # Parse JSON
            data = json.loads(response_text)

            # Convert to ReceiptData
            items = []
            for item_data in data.get('items', []):
                items.append(ReceiptItem(
                    item_name=item_data.get('item_name', ''),
                    quantity=float(item_data.get('quantity', 1.0)),
                    price=float(item_data['price']) if item_data.get('price') is not None else None,
                    category=None  # No category from Gemini (no hardcoded data)
                ))

            receipt_data = ReceiptData(
                raw_text=response_text,
                store_name=data.get('store_name'),
                date=data.get('date'),
                total_amount=float(data['total_amount']) if data.get('total_amount') is not None else None,
                subtotal=float(data['subtotal']) if data.get('subtotal') is not None else None,
                tax=float(data['tax']) if data.get('tax') is not None else None,
                items=items,
                confidence=95.0  # Gemini is highly confident
            )

            return receipt_data

        except json.JSONDecodeError as e:
            print(f"Failed to parse Gemini response as JSON: {e}")
            print(f"Response was: {response_text}")
            # Return empty receipt data
            return ReceiptData(
                raw_text=response_text,
                confidence=0.0
            )
        except Exception as e:
            print(f"Gemini OCR Error: {e}")
            return ReceiptData(
                raw_text=str(e),
                confidence=0.0
            )

    def extract_text(self, image_path: str, config: Optional[str] = None) -> Tuple[str, float]:
        """
        Legacy method for compatibility - extracts raw text only
        """
        receipt_data = self.extract_and_parse(image_path)
        return receipt_data.raw_text, receipt_data.confidence

    def get_extraction_summary(self, receipt_data: ReceiptData) -> Dict[str, Any]:
        """Get summary of extraction quality"""
        items_with_prices = sum(1 for item in receipt_data.items if item.price is not None and item.price > 0)

        return {
            'has_store_name': bool(receipt_data.store_name),
            'has_date': bool(receipt_data.date),
            'has_total': bool(receipt_data.total_amount),
            'item_count': len(receipt_data.items),
            'items_with_prices': items_with_prices,
            'confidence': receipt_data.confidence,
            'quality': 'good' if receipt_data.confidence >= 70 and items_with_prices > 0 else
                      'fair' if receipt_data.confidence >= 50 else 'poor'
        }
