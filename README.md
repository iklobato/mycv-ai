# AI Avatar Video Call Application

A production-ready AI-powered web application that simulates real-time video calls with an AI avatar that automatically embodies **Henrique Lobato's professional persona** based on his CV. Perfect for HR interviews, networking demos, or showcasing technical skills.

## 🎯 Key Features

- **🧠 CV-Based Personality**: AI avatar automatically responds as Henrique Lobato using loaded CV content
- **🎙️ Real-time Voice Interaction**: Speak naturally and get intelligent responses
- **🎭 Animated Avatar**: Lip-synced facial animation using SadTalker/Wav2Lip
- **🗣️ Voice Cloning**: AI speaks with cloned voice using XTTS-v2
- **🌐 Google Meet-style UI**: Familiar video call interface
- **🔒 100% Local**: All AI models run locally (no cloud dependencies)
- **📊 WebSocket Real-time**: Low-latency audio/video processing
- **🧪 Comprehensive Testing**: 122 tests with 90%+ coverage

## 🏗️ Architecture Overview

```
AI Avatar Application
├── Frontend (HTML5/CSS3/JS)
│   ├── WebRTC Media Capture
│   ├── WebSocket Communication
│   └── Real-time Avatar Display
├── Backend (FastAPI + Python)
│   ├── CV Service (Personality Engine)
│   ├── Whisper (Speech → Text)
│   ├── Ollama LLM (Text → Response + CV Context)
│   ├── XTTS-v2 (Text → Speech)
│   └── SadTalker (Speech → Animated Avatar)
└── AI Models (Local)
    ├── Whisper (base/small/medium)
    ├── Llama3/Mistral/Phi-3
    ├── XTTS-v2 Voice Models
    └── SadTalker Animation
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** (with uv package manager)
- **Docker & Docker Compose**
- **NVIDIA GPU** (recommended, 8GB+ VRAM)
- **16GB+ RAM** and **50GB+ free disk space**

### 1. Installation & Setup

```bash
# Clone repository
git clone <repository-url>
cd mycv-ai

# Install UV package manager
pip install uv

# Install dependencies
uv sync --extra dev

# Verify installation
uv run python --version
```

### 2. CV Configuration

```bash
# The CV is auto-created at data/cv.txt with Henrique's information
# View current CV:
cat data/cv.txt

# Edit if needed:
nano data/cv.txt
```

### 3. AI Models Setup

```bash
# Start Ollama service
ollama serve

# Pull required LLM model
ollama pull llama3

# Download other AI models (optional - for full features)
python models/download_models.py
```

### 4. Start Application

**Option A: Quick Start Script**
```bash
# Automated startup with all checks
./start_cv_avatar.sh
```

**Option B: Manual Start**
```bash
# Start FastAPI server
cd backend
uv run python main.py

# Or with uvicorn
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Option C: Docker**
```bash
# Development
docker-compose up --build

# Production
docker-compose -f docker-compose.prod.yml up
```

### 5. Access Application

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **CV Status**: http://localhost:8000/cv
- **Health Check**: http://localhost:8000/health

## 🧠 CV-Based Personality System

### How It Works

1. **Automatic Loading**: System reads `data/cv.txt` on startup
2. **Context Injection**: Every LLM request includes CV as system prompt
3. **Professional Responses**: AI answers based on Henrique's actual experience
4. **Consistent Personality**: Maintains professional persona across conversations

### System Prompt Example
```
You are Henrique Lobato, a Senior Python Developer & AI Specialist.

=== CV START ===
[Full CV content automatically inserted]
=== CV END ===

Respond as this person based on the CV information above.
Keep responses conversational and appropriate for voice interaction.
```

### CV Management

```bash
# Check CV status
curl http://localhost:8000/cv

# Reload CV after editing
curl -X POST http://localhost:8000/cv/reload

# View CV integration in prompts
curl http://localhost:8000/cv | jq '.system_prompt_preview'
```

## 📡 API Reference

### Core Endpoints

#### Text Processing
```bash
# Generate AI response (with CV context)
POST /respond
{
  "message": "Tell me about your Python experience",
  "conversation_id": "optional-session-id",
  "temperature": 0.7
}
```

#### Audio Processing
```bash
# Speech to text
POST /transcribe
Content-Type: multipart/form-data
audio_file: <wav/mp3 file>

# Text to speech
POST /speak
{
  "text": "Hello, I'm Henrique Lobato",
  "voice": "default"
}

# Generate animated avatar
POST /animate
{
  "audio_path": "/path/to/audio.wav",
  "image_path": "/path/to/photo.jpg"
}
```

#### CV Management
```bash
# Get CV information
GET /cv

# Reload CV from file
POST /cv/reload

# Check all model status
GET /models/status
```

#### Real-time Communication
```javascript
// WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws');

// Send audio for processing
ws.send(JSON.stringify({
  type: 'audio',
  data: base64AudioData
}));

// Receive animated avatar
ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  if (response.type === 'avatar_video') {
    playVideo(response.data);
  }
};
```

## 🧪 Testing

### Test Suite Overview
- **122 total tests** with **90%+ coverage**
- **Unit tests**: Individual service testing
- **Integration tests**: End-to-end CV+LLM workflows
- **API tests**: FastAPI endpoint validation

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=backend --cov-report=html

# Run specific test categories
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only
uv run pytest tests/unit/test_cv_service.py  # Specific service

# Quick CV-only test (no dependencies)
python test_cv_only.py
```

### Test Categories

#### CV Service Tests (33 tests)
- CV loading and parsing
- System prompt generation
- File operations and error handling
- Statistics calculation

#### LLM Service Tests (47 tests)
- Ollama integration
- Streaming responses
- CV context injection
- Conversation management

#### Main Application Tests (35 tests)
- API endpoints
- WebSocket handling
- Service orchestration
- Error handling

#### Integration Tests (7 tests)
- End-to-end CV+LLM workflow
- Real file operations
- Service communication

## 🛠️ Development Guide

### Project Structure

```
mycv-ai/
├── backend/                   # FastAPI application
│   ├── main.py               # FastAPI app & routes
│   ├── config.py             # Configuration settings
│   ├── services/             # Core services
│   │   ├── cv_service.py     # CV management
│   │   ├── llm_service.py    # LLM integration
│   │   ├── transcription_service.py  # Speech-to-text
│   │   ├── tts_service.py    # Text-to-speech
│   │   └── animation_service.py      # Avatar animation
│   └── models/               # Pydantic schemas
├── frontend/                 # Web interface
│   ├── index.html           # Main UI
│   ├── js/                  # JavaScript modules
│   └── styles/              # CSS styling
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── conftest.py         # Test configuration
├── data/                    # Application data
│   └── cv.txt              # Henrique's CV
├── models/                  # AI model storage
├── avatar_photos/           # Avatar images
├── voice_samples/           # Voice training data
└── temp/                    # Temporary files
```

### Development Workflow

```bash
# Setup development environment
uv sync --extra dev

# Start development server with hot reload
cd backend
uv run uvicorn main:app --reload --log-level debug

# Run linting and formatting
uv run black backend tests
uv run isort backend tests
uv run flake8 backend

# Run tests in watch mode
uv run pytest --ff -x

# Generate test coverage
uv run pytest --cov=backend --cov-report=html
open htmlcov/index.html
```

### Adding New Features

1. **Update CV Service** for personality changes
2. **Extend LLM Service** for new conversation features  
3. **Add API Endpoints** in `main.py`
4. **Write Tests** in appropriate test files
5. **Update Configuration** in `config.py`
6. **Document Changes** in this README

### Configuration Management

```python
# Environment-based settings
export ENVIRONMENT=development  # or production
export OLLAMA_BASE_URL=http://localhost:11434
export LLM_MODEL=llama3
export LOG_LEVEL=DEBUG

# Or use .env file
echo "ENVIRONMENT=development" > .env
echo "DEBUG=true" >> .env
```

## 🎭 Usage Examples

### Professional Interview Simulation

```javascript
// Ask technical questions
const response = await fetch('/respond', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    message: "What's your experience with Django and FastAPI?",
    conversation_id: "interview-session"
  })
});

// Expected response (as Henrique):
// "I have extensive experience with both frameworks. I've been working 
//  with Django for over 6 years and FastAPI for 3 years. In my current 
//  role at AI Avatar Systems, I use FastAPI for building high-performance 
//  APIs..."
```

### Skills Assessment

```bash
# Technical skills inquiry
curl -X POST http://localhost:8000/respond \
  -H "Content-Type: application/json" \
  -d '{"message": "What programming languages do you know?"}'

# Project experience
curl -X POST http://localhost:8000/respond \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about your AI Avatar project"}'
```

### Real-time Conversation

```javascript
// WebSocket conversation
const ws = new WebSocket('ws://localhost:8000/ws');

// Send voice message
navigator.mediaDevices.getUserMedia({audio: true})
  .then(stream => {
    const recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (event) => {
      ws.send(JSON.stringify({
        type: 'audio',
        data: arrayBufferToBase64(event.data)
      }));
    };
  });

// Receive avatar response
ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  if (response.type === 'avatar_response') {
    document.getElementById('avatar-video').src = response.video_url;
  }
};
```

## 🔧 Configuration Options

### Environment Variables

```bash
# Core settings
APP_NAME="AI Avatar Application"
DEBUG=false
HOST=0.0.0.0
PORT=8000

# AI Models
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3
WHISPER_MODEL=base
TTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2

# Performance
MAX_CONCURRENT_REQUESTS=10
USE_GPU=true
GPU_MEMORY_FRACTION=0.8

# Security
ALLOWED_HOSTS=["localhost", "127.0.0.1"]
MAX_FILE_SIZE=52428800  # 50MB
```

### Model Configuration

```python
# In config.py
MODELS_CONFIG = {
    "ollama": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "num_ctx": 2048,
        "num_predict": 512
    },
    "whisper": {
        "model_size": "base",
        "compute_type": "float16",
        "device": "auto"
    },
    "xtts": {
        "temperature": 0.75,
        "top_p": 0.85,
        "length_penalty": 1.0
    }
}
```

## 🐳 Docker Deployment

### Development

```bash
# Start all services
docker-compose up --build

# View logs
docker-compose logs -f app

# Scale services
docker-compose up --scale app=2
```

### Production

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Health check
docker-compose ps
docker-compose exec app curl http://localhost:8000/health
```

### Docker Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    depends_on:
      - ollama
      
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
```

## 🚨 Troubleshooting

### Common Issues

#### CV Not Loading
```bash
# Check CV file
ls -la data/cv.txt
cat data/cv.txt | wc -l

# Test CV service
python test_cv_only.py

# Check CV API
curl http://localhost:8000/cv | jq '.cv_info.has_cv'
```

#### Ollama Connection Issues
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve

# Re-pull model
ollama pull llama3
```

#### Performance Issues
```bash
# Check GPU usage
nvidia-smi

# Monitor memory
free -h
df -h

# Check logs
tail -f backend/logs/app.log
```

#### Audio/Video Issues
```bash
# Test audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Check WebRTC support
# Open browser console and check for getUserMedia errors
```

### Debugging Tools

```bash
# Debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Test specific service
python -c "from backend.services.cv_service import CVService; print(CVService().get_cv_info())"

# Validate configuration
python -c "from backend.config import settings; print(settings.dict())"
```

## 📊 Performance Optimization

### GPU Optimization
```python
# Adjust batch sizes for your GPU
MODELS_CONFIG = {
    "whisper": {"batch_size": 16},  # Reduce if OOM
    "xtts": {"batch_size": 1},      # Keep at 1 for real-time
    "sadtalker": {"batch_size": 1}  # Keep at 1 for real-time
}
```

### Memory Management
```bash
# Monitor memory usage
watch -n 1 'free -h && nvidia-smi'

# Clear cache periodically
curl -X POST http://localhost:8000/admin/clear_cache
```

### Network Optimization
```javascript
// Compress audio before sending
const compressedAudio = await compressAudio(audioBlob, {
  bitrate: 64000,
  sampleRate: 16000
});
```

## 🤝 Contributing

### Development Setup
```bash
# Fork and clone
git clone your-fork-url
cd mycv-ai

# Create development branch
git checkout -b feature/your-feature

# Install development dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Make changes and test
# Submit pull request
```

### Code Quality Standards
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **pytest** for testing (90%+ coverage required)
- **Type hints** for all public functions
- **Docstrings** for all classes and methods

## 📄 License

MIT License - see LICENSE file for details.

## 🙋‍♂️ Support

- **Documentation**: Check this README and API docs at `/docs`
- **Issues**: Report bugs via GitHub issues
- **Testing**: Run `python test_cv_only.py` for quick diagnostics
- **Health Check**: Visit `/health` endpoint for system status

---

**Built with ❤️ for AI-powered professional interactions** 