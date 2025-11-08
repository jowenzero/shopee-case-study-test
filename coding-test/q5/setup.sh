#!/bin/bash

echo "Setting up Food Receipt AI Platform..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file from example if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env and add your GEMINI_API_KEY"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Edit .env and add your GEMINI_API_KEY"
echo "3. Run the app: streamlit run app.py"
