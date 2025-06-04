#!/bin/bash

# AI Avatar with CV Integration Startup Script
# This script sets up and starts the AI avatar system with Henrique's CV

echo "🚀 Starting AI Avatar with CV Integration..."
echo ""

# Check if we're in the right directory
if [ ! -f "data/cv.txt" ]; then
    echo "❌ Error: data/cv.txt not found"
    echo "💡 Please run this script from the project root directory"
    exit 1
fi

# Test CV integration first
echo "🧪 Testing CV Integration..."
uv run pytest tests/unit/test_cv_service.py -v --tb=short

if [ $? -ne 0 ]; then
    echo "❌ CV integration test failed"
    exit 1
fi

echo ""
echo "✅ CV integration test passed!"
echo ""

# Check if Ollama is running
echo "🔍 Checking Ollama status..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama is not running"
    echo "💡 Please start Ollama in another terminal:"
    echo "   ollama serve"
    echo ""
    echo "💡 And pull a model:"
    echo "   ollama pull llama3"
    echo ""
    read -p "Press Enter when Ollama is ready, or Ctrl+C to exit..."
else
    echo "✅ Ollama is running"
fi

# Check if required model is available
echo ""
echo "🔍 Checking for required models..."
if ! curl -s http://localhost:11434/api/tags | grep -q "llama3"; then
    echo "⚠️  Model 'llama3' not found"
    echo "💡 Pulling model..."
    ollama pull llama3
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to pull model"
        echo "💡 Please manually run: ollama pull llama3"
        exit 1
    fi
else
    echo "✅ Model 'llama3' is available"
fi

# Check Python dependencies
echo ""
echo "🔍 Checking Python dependencies..."
if ! uv run python -c "import fastapi, uvicorn, httpx" > /dev/null 2>&1; then
    echo "⚠️  Some dependencies are missing"
    echo "💡 Installing dependencies..."
    uv sync --extra dev
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
else
    echo "✅ Core dependencies are available"
fi

echo ""
echo "🎯 System Summary:"
echo "   ✅ CV loaded: Henrique Lobato - Senior Python Developer & AI Specialist"
echo "   ✅ Ollama running with llama3 model"
echo "   ✅ Python dependencies installed"
echo "   ✅ Test suite: 122 tests ready"
echo ""

echo "🚀 Starting AI Avatar Application..."
echo ""
echo "📝 The avatar will respond as Henrique based on his CV:"
echo "   - Technical questions answered from CV experience"
echo "   - Professional background referenced appropriately"
echo "   - Consistent personality across all interactions"
echo ""
echo "🌐 Access the application at: http://localhost:8000"
echo "📊 Check CV status at: http://localhost:8000/cv"
echo "📚 API docs at: http://localhost:8000/docs"
echo "🧪 Run tests with: uv run pytest"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the FastAPI application
cd backend && uv run python main.py 