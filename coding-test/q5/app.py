import streamlit as st
import os
import tempfile
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv

from src.ocr_extractor import OCRExtractor
from src.storage_integration import StorageIntegration

load_dotenv()

st.set_page_config(
    page_title="Food Receipt AI Platform",
    page_icon="🧾",
    layout="wide"
)

st.title("Food Receipt AI Platform")
st.markdown("Upload food receipts and extract data using AI")

@st.cache_resource
def initialize_components():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        st.error("GEMINI_API_KEY not found in .env file")
        st.stop()

    ocr = OCRExtractor()
    storage = StorageIntegration(
        db_path="./data/receipts.db",
        vector_db_path="./data/vector_db.json"
    )
    return ocr, storage

ocr_extractor, storage_integration = initialize_components()

tabs = st.tabs(["Upload Receipt", "View Receipts"])

with tabs[0]:
    st.header("Upload Receipt")

    # Initialize file uploader key in session state
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0

    uploaded_file = st.file_uploader(
        "Choose a receipt image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        help="Drag and drop or click to browse",
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )
    st.session_state.skip_extraction = False

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Receipt Image")
            image = Image.open(uploaded_file)
            st.image(image, width='stretch')

        with col2:
            st.subheader("Extracted Data")

            if ('extracted_data' not in st.session_state or st.session_state.get('current_file') != uploaded_file.name) and st.session_state.skip_extraction == False:
                with st.spinner("Extracting data from receipt..."):
                    temp_path = None
                    try:
                        # Save to temporary file for OCR processing
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            image.save(tmp_file.name)
                            temp_path = tmp_file.name

                        receipt_data = ocr_extractor.extract_and_parse(temp_path)
                        st.session_state.extracted_data = receipt_data
                        st.session_state.current_file = uploaded_file.name
                        st.session_state.uploaded_image = image

                        st.success("Data extracted successfully!")
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")
                        st.stop()
                    finally:
                        # Always clean up temp file, even if there's an error
                        if temp_path and os.path.exists(temp_path):
                            os.unlink(temp_path)

            receipt_data = st.session_state.extracted_data

            st.markdown("### Receipt Information")

            store_name = st.text_input("Store Name", value=receipt_data.store_name or "", key="store_name")
            date = st.date_input("Date", value=datetime.strptime(receipt_data.date, "%Y-%m-%d").date() if receipt_data.date else datetime.now().date(), key="date")
            total_amount = st.number_input("Total Amount (Rp)", value=float(receipt_data.total_amount) if receipt_data.total_amount else 0.0, key="total")

            if receipt_data.subtotal:
                subtotal = st.number_input("Subtotal (Rp)", value=float(receipt_data.subtotal), key="subtotal")

            if receipt_data.tax:
                tax = st.number_input("Tax/Service (Rp)", value=float(receipt_data.tax), key="tax")

            st.markdown("### Items")

            if receipt_data.items:
                items_data = []
                for i, item in enumerate(receipt_data.items):
                    with st.expander(f"Item {i+1}: {item.item_name}", expanded=True):
                        item_name = st.text_input(
                            "Item Name",
                            value=item.item_name,
                            key=f"item_name_{i}"
                        )

                        col_qty, col_price, col_delete = st.columns([2, 2, 1])
                        with col_qty:
                            quantity = st.number_input(
                                "Quantity",
                                value=float(item.quantity),
                                key=f"qty_{i}"
                            )
                        with col_price:
                            price = st.number_input(
                                "Price (Rp)",
                                value=float(item.price) if item.price else 0.0,
                                key=f"price_{i}",
                                help="Negative values = discounts"
                            )
                        with col_delete:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button("Delete", key=f"delete_{i}", type="secondary"):
                                if 'items_to_delete' not in st.session_state:
                                    st.session_state.items_to_delete = set()
                                st.session_state.items_to_delete.add(i)
                                st.rerun()

                        items_data.append({
                            'item_name': item_name,
                            'quantity': quantity,
                            'price': price
                        })

                if 'items_to_delete' in st.session_state and st.session_state.items_to_delete:
                    items_data = [item for idx, item in enumerate(items_data) if idx not in st.session_state.items_to_delete]
                    receipt_data.items = [item for idx, item in enumerate(receipt_data.items) if idx not in st.session_state.items_to_delete]
                    st.session_state.extracted_data = receipt_data
                    st.session_state.items_to_delete = set()
                    st.rerun()

                st.session_state.items_data = items_data
            else:
                st.warning("No items detected. You can add items manually below.")
                st.session_state.items_data = []

            add_item = st.button("Add Item")
            if add_item:
                if 'items_data' not in st.session_state:
                    st.session_state.items_data = []
                st.session_state.items_data.append({
                    'item_name': '',
                    'quantity': 1.0,
                    'price': 0.0
                })
                st.rerun()

            st.markdown("---")

            col_save, col_cancel = st.columns(2)

            with col_save:
                if st.button("Save to Database", type="primary", width='stretch'):
                    with st.spinner("Saving to database..."):
                        try:
                            from src.ocr_extractor import ReceiptData, ReceiptItem

                            items = []
                            for item_dict in st.session_state.get('items_data', []):
                                if item_dict['item_name']:
                                    items.append(ReceiptItem(
                                        item_name=item_dict['item_name'],
                                        quantity=item_dict['quantity'],
                                        price=item_dict['price'],
                                        category=None
                                    ))

                            updated_receipt = ReceiptData(
                                store_name=st.session_state.store_name,
                                date=st.session_state.date.strftime("%Y-%m-%d"),
                                total_amount=st.session_state.total,
                                subtotal=st.session_state.get('subtotal'),
                                tax=st.session_state.get('tax'),
                                items=items,
                                confidence=receipt_data.confidence,
                                raw_text=receipt_data.raw_text
                            )
                            
                            result = storage_integration.store_receipt(updated_receipt)

                            if result['success']:
                                st.success(f"Receipt saved successfully! Receipt ID: {result['receipt_id']}")
                                st.balloons()

                                st.info(f"Stored {result['item_count']} items in database")

                                # Clean up session state
                                del st.session_state.extracted_data
                                del st.session_state.current_file
                                del st.session_state.items_data
                                if 'uploaded_image' in st.session_state:
                                    del st.session_state.uploaded_image
                            else:
                                st.error(f"Failed to save: {result.get('error', 'Unknown error')}")

                        except Exception as e:
                            st.error(f"Save failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())

            with col_cancel:
                if st.button("Clear", width='stretch'):
                    if 'extracted_data' in st.session_state:
                        del st.session_state.extracted_data
                    if 'current_file' in st.session_state:
                        del st.session_state.current_file
                    if 'items_data' in st.session_state:
                        del st.session_state.items_data
                    if 'uploaded_image' in st.session_state:
                        del st.session_state.uploaded_image

                    st.session_state.file_uploader_key += 1
                    st.session_state.skip_extraction = True
                    st.rerun()

with tabs[1]:
    st.header("Stored Receipts")

    try:
        stats = storage_integration.get_statistics()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Receipts", stats['total_receipts'])
        with col2:
            st.metric("Total Items", stats['total_items'])
        with col3:
            st.metric("Total Vectors", stats['total_vectors'])

        st.markdown("---")

        receipts = storage_integration.db.get_all_receipts()

        if receipts:
            for receipt in receipts:
                with st.expander(f"{receipt['store_name']} - {receipt['upload_date']} - Rp{receipt['total_amount']:,.0f}" if receipt['total_amount'] else f"{receipt['store_name']} - {receipt['upload_date']}"):
                    st.markdown(f"**Receipt ID:** {receipt['id']}")
                    st.markdown(f"**Store:** {receipt['store_name']}")
                    st.markdown(f"**Date:** {receipt['upload_date']}")
                    if receipt['total_amount']:
                        st.markdown(f"**Total:** Rp{receipt['total_amount']:,.0f}")

                    full_receipt = storage_integration.get_receipt_with_context(receipt['id'])
                    if full_receipt and full_receipt.get('items'):
                        st.markdown("**Items:**")
                        for item in full_receipt['items']:
                            price_str = f"Rp{item['price']:,.0f}" if item['price'] else "N/A"
                            qty_str = f"{item['quantity']}x" if item['quantity'] != 1.0 else ""
                            st.markdown(f"- {qty_str} {item['item_name']}: {price_str}")
        else:
            st.info("No receipts stored yet. Upload a receipt to get started!")

    except Exception as e:
        st.error(f"Error loading receipts: {e}")

st.sidebar.title("About")
st.sidebar.info(
    """
    This AI-powered platform allows you to:
    - Upload food receipt images
    - Extract data using Gemini Vision API
    - Store receipts in SQLite and Vector DB
    - Query receipts using natural language
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack:**")
st.sidebar.markdown("- Gemini 2.5 Flash (Vision & LLM)")
st.sidebar.markdown("- LangChain + LangGraph")
st.sidebar.markdown("- SQLite + Vector DB")
st.sidebar.markdown("- Sentence Transformers")
st.sidebar.markdown("- Streamlit")
