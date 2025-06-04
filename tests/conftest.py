"""
Shared pytest fixtures for AI Avatar tests.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_cv_content() -> str:
    """Sample CV content for testing."""
    return """Henrique Lobato
Senior Python Developer

Email: henrique@example.com
Phone: +1234567890
Location: São Paulo, Brazil

PROFESSIONAL SUMMARY
Experienced Python developer with 8+ years in web development, AI systems, and cloud architecture.
Specialized in FastAPI, Django, and machine learning applications.

TECHNICAL SKILLS
- Programming Languages: Python, JavaScript, TypeScript
- Web Frameworks: FastAPI, Django, Flask, React
- Databases: PostgreSQL, MySQL, MongoDB, Redis
- Cloud Platforms: AWS, Google Cloud, Azure
- AI/ML: TensorFlow, PyTorch, scikit-learn, Transformers
- DevOps: Docker, Kubernetes, CI/CD, Linux

EXPERIENCE

Senior Python Developer | TechCorp | 2020 - Present
- Lead development of AI-powered applications using FastAPI and PyTorch
- Designed and implemented scalable microservices architecture
- Mentored junior developers and established coding standards
- Improved system performance by 40% through optimization

Python Developer | StartupXYZ | 2018 - 2020
- Built web applications using Django and React
- Implemented RESTful APIs and integrated third-party services
- Worked with agile development methodologies
- Contributed to open-source projects

EDUCATION
Bachelor of Computer Science | University of São Paulo | 2016
Some basic experience with Python and web development."""


@pytest.fixture
def minimal_cv_content() -> str:
    """Minimal CV content for testing."""
    return """John Doe
Python Developer

Some basic experience with Python and web development."""


@pytest.fixture
def invalid_cv_content() -> str:
    """Invalid CV content for testing."""
    return "Too short"


@pytest.fixture
def temp_cv_file(sample_cv_content: str) -> Generator[Path, None, None]:
    """Create a temporary CV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_cv_content)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_empty_cv_file() -> Generator[Path, None, None]:
    """Create a temporary empty CV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("")
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_cv_service():
    """Mock CV service instance."""
    mock_service = MagicMock()
    mock_service.is_ready.return_value = True
    mock_service.has_cv.return_value = True
    mock_service.get_cv_content.return_value = "Sample CV content"
    mock_service.get_system_prompt.return_value = "You are Henrique Lobato..."
    mock_service.get_cv_info.return_value = {
        "has_cv": True,
        "default_name": "Henrique Lobato",
        "default_title": "Senior Python Developer",
        "content_length": 1000,
        "last_loaded": "2024-01-15T10:30:00",
        "file_exists": True,
        "content_preview": "HENRIQUE LOBATO..."
    }
    mock_service.get_cv_stats.return_value = {
        "character_count": 1000,
        "word_count": 150,
        "line_count": 30,
        "paragraph_count": 8
    }
    mock_service.reload_cv = AsyncMock(return_value=True)
    mock_service.initialize = AsyncMock()
    mock_service.cleanup = AsyncMock()
    return mock_service


@pytest.fixture
def mock_httpx_client():
    """Mock HTTPX async client."""
    mock_client = AsyncMock()
    mock_client.get.return_value = MagicMock(status_code=200)
    mock_client.post.return_value = MagicMock(status_code=200)
    mock_client.stream.return_value.__aenter__.return_value = AsyncMock(status_code=200)
    mock_client.aclose = AsyncMock()
    return mock_client


@pytest.fixture
def sample_llm_request():
    """Sample LLM request for testing."""
    from backend.models.schemas import LLMRequest
    return LLMRequest(
        message="What is your Python experience?",
        conversation_id="test-conversation-123",
        temperature=0.7,
        max_tokens=150
    )


@pytest.fixture
def sample_ollama_models_response():
    """Sample Ollama models API response."""
    return {
        "models": [
            {
                "name": "llama3:latest",
                "modified_at": "2024-01-15T10:30:00Z",
                "size": 4368491520,
                "digest": "abc123"
            },
            {
                "name": "mistral:latest", 
                "modified_at": "2024-01-14T15:20:00Z",
                "size": 4109086720,
                "digest": "def456"
            },
            {
                "name": "phi3:latest",
                "modified_at": "2024-01-13T09:15:00Z", 
                "size": 2311071744,
                "digest": "ghi789"
            }
        ]
    }


@pytest.fixture
def sample_ollama_chat_response():
    """Sample Ollama chat API response."""
    return {
        "message": {
            "content": "I have over 8 years of Python experience, working extensively with Django, FastAPI, and Flask."
        },
        "done": True
    }


@pytest.fixture
def mock_aiofiles_open():
    """Mock aiofiles.open for file operations."""
    with patch('aiofiles.open') as mock_open:
        yield mock_open


@pytest.fixture
def mock_path_exists():
    """Mock Path.exists() method."""
    with patch('pathlib.Path.exists') as mock_exists:
        mock_exists.return_value = True
        yield mock_exists


@pytest.fixture
def mock_path_mkdir():
    """Mock Path.mkdir() method."""
    with patch('pathlib.Path.mkdir') as mock_mkdir:
        yield mock_mkdir


@pytest.fixture
def fastapi_test_client():
    """FastAPI test client with mocked dependencies."""
    with patch('backend.main.cv_service') as mock_cv, \
         patch('backend.main.transcription_service') as mock_trans, \
         patch('backend.main.llm_service') as mock_llm, \
         patch('backend.main.tts_service') as mock_tts, \
         patch('backend.main.animation_service') as mock_anim, \
         patch('backend.main.websocket_manager') as mock_ws:
        
        # Mock all services as ready
        mock_cv.is_ready.return_value = True
        mock_cv.initialize = AsyncMock(return_value=True)
        mock_cv.cleanup = AsyncMock()
        mock_trans.is_ready.return_value = True
        mock_trans.initialize = AsyncMock(return_value=True)
        mock_trans.cleanup = AsyncMock()
        mock_llm.is_ready.return_value = True
        mock_llm.initialize = AsyncMock(return_value=True)
        mock_llm.cleanup = AsyncMock()
        mock_tts.is_ready.return_value = True
        mock_tts.initialize = AsyncMock(return_value=True)
        mock_tts.cleanup = AsyncMock()
        mock_anim.is_ready.return_value = True
        mock_anim.initialize = AsyncMock(return_value=True)
        mock_anim.cleanup = AsyncMock()
        
        # Import app after mocking
        from backend.main import app
        
        with TestClient(app) as client:
            yield client


@pytest.fixture
def mock_websocket():
    """Mock WebSocket for testing."""
    mock_ws = AsyncMock()
    mock_ws.receive.return_value = {"type": "websocket.receive", "text": '{"type": "ping"}'}
    mock_ws.send_json = AsyncMock()
    mock_ws.send_bytes = AsyncMock()
    return mock_ws


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    with patch('logging.getLogger') as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        yield mock_logger


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Mark all tests as offline by default
def pytest_collection_modifyitems(config, items):
    """Add offline marker to all tests."""
    for item in items:
        item.add_marker(pytest.mark.offline) 


@pytest.fixture
def mock_file_system():
    """Mock file system operations."""
    with patch('pathlib.Path.exists') as mock_exists, \
         patch('pathlib.Path.read_text') as mock_read, \
         patch('pathlib.Path.write_text') as mock_write, \
         patch('pathlib.Path.mkdir') as mock_mkdir:
        
        mock_exists.return_value = True
        mock_read.return_value = "Mock file content"
        
        yield {
            'exists': mock_exists,
            'read_text': mock_read,
            'write_text': mock_write,
            'mkdir': mock_mkdir
        }


@pytest.fixture
def mock_settings():
    """Mock application settings."""
    mock_settings = MagicMock()
    mock_settings.WHISPER_MODEL = "base"
    mock_settings.LLM_MODEL = "llama3"
    mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
    mock_settings.VOICE_MODEL_PATH = "./models/voice"
    mock_settings.AVATAR_MODEL_PATH = "./models/avatar"
    mock_settings.MODELS_CONFIG = {
        "ollama": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_ctx": 2048,
            "num_predict": 150
        }
    }
    return mock_settings


@pytest.fixture(autouse=True)
def isolate_tests():
    """Ensure test isolation by resetting global state."""
    yield
    # Reset any global state if needed


# Async test utilities
@pytest.fixture
def async_mock():
    """Factory for creating async mocks."""
    def _create_async_mock(*args, **kwargs):
        return AsyncMock(*args, **kwargs)
    return _create_async_mock


# Error simulation fixtures
@pytest.fixture
def connection_error():
    """HTTP connection error for testing."""
    return httpx.ConnectError("Connection failed")


@pytest.fixture
def http_error():
    """HTTP error for testing."""
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 500
    return httpx.HTTPStatusError("Server error", request=mock_request, response=mock_response)


# Test data fixtures
@pytest.fixture
def sample_audio_data():
    """Sample audio data for testing."""
    return b"fake_audio_data_for_testing" * 100


@pytest.fixture
def sample_image_data():
    """Sample image data for testing."""
    return b"fake_image_data_" + b"0" * 2000  # 2KB of fake image


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history for testing."""
    return [
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I'm Henrique Lobato, a Senior Python Developer."},
        {"role": "user", "content": "What's your experience with Django?"},
        {"role": "assistant", "content": "I have 6+ years of Django experience..."}
    ]


@pytest.fixture
def large_conversation_history():
    """Large conversation history for testing truncation."""
    history = []
    for i in range(30):  # More than max allowed
        history.append({"role": "user", "content": f"Question {i}"})
        history.append({"role": "assistant", "content": f"Answer {i}"})
    return history 