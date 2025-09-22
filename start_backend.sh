#!/bin/bash

# Deep Researcher Agent - Backend Startup Script

echo "🚀 Starting Deep Researcher Agent Backend..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ and try again."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version+ is required. Current version: $python_version"
    exit 1
fi

# Navigate to backend directory
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating example configuration..."
    cat > .env << EOF
# Deep Researcher Agent - Environment Configuration

# OpenAI API Configuration (Required for full functionality)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Alternative: HuggingFace Configuration (if not using OpenAI)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
HUGGINGFACE_MODEL=microsoft/DialoGPT-medium

# Vector Store Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE_PATH=./data/vector_store
MAX_DOCUMENT_SIZE=10000000

# Document Processing
SUPPORTED_FORMATS=txt,pdf,docx
MAX_DOCUMENTS_PER_QUERY=50

# Report Generation
REPORT_OUTPUT_DIR=./data/reports
DEFAULT_REPORT_FORMAT=pdf

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
EOF
    echo "📝 Please edit .env file and add your OpenAI API key for full functionality."
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/documents data/vector_store data/reports

# Start the backend server
echo "🌟 Starting Deep Researcher Agent Backend..."
echo "📡 API will be available at: http://localhost:8000"
echo "📖 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py


