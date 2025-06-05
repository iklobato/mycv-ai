"""
Unit tests for the TranscriptionService.

Tests cover:
- Service initialization and configuration
- Audio transcription functionality  
- Error handling and edge cases
- Device selection and model loading
- Audio preprocessing
- Streaming transcription
- Language detection
"""

import pytest
import tempfile
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

# Import the service (mocks are set up in conftest.py)
from backend.services.transcription_service import TranscriptionService
from backend.models.schemas import TranscriptionRequest, TranscriptionResponse


class TestTranscriptionService:
    """Test suite for TranscriptionService."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.service = TranscriptionService()
    
    def teardown_method(self):
        """Clean up after each test."""
        # Clean up any resources
        pass
    
    def test_transcription_service_initialization(self):
        """Test TranscriptionService initialization."""
        service = TranscriptionService()
        
        assert service.model is None
        assert service.fast_model is None
        assert service.device is None
        assert service.is_initialized is False
        assert service.model_type == "faster_whisper"
    
    async def test_initialize_success(self):
        """Test successful service initialization."""
        with patch('backend.services.transcription_service.settings') as mock_settings:
            mock_settings.MODELS_CONFIG = {
                "whisper": {
                    "compute_type": "int8",
                    "num_workers": 1
                }
            }
            mock_settings.WHISPER_MODEL = "base"
            mock_settings.WHISPER_DEVICE = "cpu"
            mock_settings.USE_GPU = False
            
            await self.service.initialize()
            
            assert self.service.is_initialized is True
            assert self.service.device == "cpu"
    
    async def test_initialize_failure(self):
        """Test service initialization failure."""
        with patch('backend.services.transcription_service.settings') as mock_settings:
            mock_settings.MODELS_CONFIG = {
                "whisper": {
                    "compute_type": "int8", 
                    "num_workers": 1
                }
            }
            mock_settings.WHISPER_MODEL = "base"
            mock_settings.WHISPER_DEVICE = "cpu"
            mock_settings.USE_GPU = False
            
            # Mock faster whisper to fail
            with patch('backend.services.transcription_service.WhisperModel', side_effect=Exception("Model load failed")):
                # Also mock openai whisper to fail
                with patch('backend.services.transcription_service.whisper.load_model', side_effect=Exception("OpenAI model failed")):
                    with pytest.raises(Exception, match="OpenAI model failed"):
                        await self.service.initialize()
    
    def test_get_device_auto_cpu(self):
        """Test device selection with auto setting (CPU)."""
        with patch('backend.services.transcription_service.settings') as mock_settings:
            mock_settings.WHISPER_DEVICE = "auto"
            mock_settings.USE_GPU = False
            
            device = self.service._get_device()
            assert device == "cpu"
    
    def test_get_device_specific(self):
        """Test device selection with specific device."""
        with patch('backend.services.transcription_service.settings') as mock_settings:
            mock_settings.WHISPER_DEVICE = "cuda"
            
            device = self.service._get_device()
            assert device == "cuda"
    
    async def test_transcribe_success_faster_whisper(self):
        """Test successful transcription with Faster Whisper."""
        # Initialize service
        self.service.is_initialized = True
        self.service.model_type = "faster_whisper"
        self.service.fast_model = Mock()
        
        # Mock audio preprocessing
        mock_audio = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        with patch.object(self.service, '_preprocess_audio', return_value=mock_audio) as mock_preprocess, \
             patch.object(self.service, '_transcribe_faster_whisper') as mock_transcribe:
            
            mock_transcribe.return_value = {
                "text": "Hello world",
                "confidence": 0.95,
                "language": "en",
                "duration": 2.5
            }
            
            request = TranscriptionRequest(
                audio_data=b"fake_audio_data",
                language="en"
            )
            
            response = await self.service.transcribe(request)
            
            assert response.success is True
            assert response.text == "Hello world"
            assert response.confidence == 0.95
            assert response.language == "en"
            assert response.processing_time > 0
            
            mock_preprocess.assert_called_once_with(b"fake_audio_data")
            mock_transcribe.assert_called_once()
    
    async def test_transcribe_success_openai_whisper(self):
        """Test successful transcription with OpenAI Whisper."""
        # Initialize service  
        self.service.is_initialized = True
        self.service.model_type = "openai_whisper"
        self.service.model = Mock()
        self.service.fast_model = None
        
        mock_audio = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        with patch.object(self.service, '_preprocess_audio', return_value=mock_audio) as mock_preprocess, \
             patch.object(self.service, '_transcribe_openai_whisper') as mock_transcribe:
            
            mock_transcribe.return_value = {
                "text": "Hello world",
                "confidence": 0.95,
                "language": "en",
                "duration": 2.5
            }
            
            request = TranscriptionRequest(
                audio_data=b"fake_audio_data",
                language="en"
            )
            
            response = await self.service.transcribe(request)
            
            assert response.success is True
            assert response.text == "Hello world"
            assert response.confidence == 0.95
    
    async def test_transcribe_not_initialized(self):
        """Test transcription when service not initialized."""
        request = TranscriptionRequest(
            audio_data=b"fake_audio_data",
            language="en"
        )
        
        with pytest.raises(RuntimeError, match="Transcription service not initialized"):
            await self.service.transcribe(request)
    
    async def test_transcribe_exception_handling(self):
        """Test transcription exception handling."""
        self.service.is_initialized = True
        
        with patch.object(self.service, '_preprocess_audio', side_effect=Exception("Audio processing failed")):
            request = TranscriptionRequest(
                audio_data=b"fake_audio_data",
                language="en"
            )
            
            response = await self.service.transcribe(request)
            
            assert response.success is False
            assert response.text == ""
            assert response.confidence == 0.0
            assert "Audio processing failed" in response.error_message
    
    async def test_preprocess_audio_success(self):
        """Test successful audio preprocessing."""
        audio_data = b"fake_wav_data"
        
        with patch('backend.services.transcription_service.tempfile.NamedTemporaryFile') as mock_temp, \
             patch('backend.services.transcription_service.librosa') as mock_librosa, \
             patch('backend.services.transcription_service.settings') as mock_settings:
            
            mock_settings.AUDIO_SAMPLE_RATE = 16000
            
            # Mock temporary file
            mock_file = Mock()
            mock_file.name = "/tmp/test.wav"
            mock_temp.return_value.__enter__.return_value = mock_file
            
            # Mock librosa
            mock_audio_array = [0.1, 0.2, 0.3]
            mock_librosa.load.return_value = (mock_audio_array, 16000)
            mock_librosa.util.normalize.return_value = mock_audio_array
            mock_librosa.effects.trim.return_value = (mock_audio_array, [0, len(mock_audio_array)])
            
            result = await self.service._preprocess_audio(audio_data)
            
            assert result == mock_audio_array
            mock_file.write.assert_called_once_with(audio_data)
            mock_librosa.load.assert_called_once()
    
    async def test_transcribe_faster_whisper_success(self):
        """Test Faster Whisper transcription."""
        self.service.fast_model = Mock()
        
        # Import MockSegment for proper structure
        from tests.mocks.ml_dependencies import MockSegment
        
        # Mock the transcribe method to return segments and info
        mock_segments = [
            MockSegment(0.0, 2.5, "Hello world", 0.95)
        ]
        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.98
        mock_info.duration = 2.5
        
        self.service.fast_model.transcribe.return_value = (mock_segments, mock_info)
        
        audio_array = [0.1, 0.2, 0.3]
        request = TranscriptionRequest(
            audio_data=b"fake_audio_data",
            language="en"
        )
        
        result = await self.service._transcribe_faster_whisper(audio_array, request)
        
        assert result["text"] == "Hello world"
        assert result["confidence"] == 0.98  # language_probability
        assert result["language"] == "en"
        assert result["duration"] == 2.5
    
    async def test_transcribe_openai_whisper_success(self):
        """Test OpenAI Whisper transcription."""
        self.service.model = Mock()
        
        # Mock the transcribe method
        mock_result = {
            "text": "Hello world",
            "language": "en",
            "segments": [
                {
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 2.5,
                    "avg_logprob": -0.1
                }
            ]
        }
        self.service.model.transcribe.return_value = mock_result
        
        audio_array = [0.1, 0.2, 0.3]
        request = TranscriptionRequest(
            audio_data=b"fake_audio_data",
            language="en"
        )
        
        result = await self.service._transcribe_openai_whisper(audio_array, request)
        
        assert result["text"] == "Hello world"
        assert result["language"] == "en"
    
    async def test_transcribe_streaming_success(self):
        """Test streaming transcription."""
        self.service.is_initialized = True
        self.service.model_type = "faster_whisper"
        self.service.fast_model = Mock()
        
        # Create larger audio chunks that will trigger transcription
        chunk_size = 16000 * 2 * 2 + 1000  # Slightly larger than the threshold
        audio_chunks = [b"0" * chunk_size, b"1" * chunk_size, b"2" * chunk_size]
        
        with patch.object(self.service, 'transcribe') as mock_transcribe, \
             patch('backend.services.transcription_service.settings') as mock_settings:
            
            mock_settings.AUDIO_SAMPLE_RATE = 16000
            
            # Create mock transcription responses
            from backend.models.schemas import TranscriptionResponse
            mock_responses = [
                TranscriptionResponse(
                    text="Hello",
                    language="en",
                    confidence=0.95,
                    word_count=1,
                    processing_time=0.1,
                    success=True
                ),
                TranscriptionResponse(
                    text="world",
                    language="en", 
                    confidence=0.95,
                    word_count=1,
                    processing_time=0.1,
                    success=True
                ),
                TranscriptionResponse(
                    text="test",
                    language="en",
                    confidence=0.95,
                    word_count=1,
                    processing_time=0.1,
                    success=True
                )
            ]
            
            mock_transcribe.side_effect = mock_responses
            
            results = []
            async for text in self.service.transcribe_streaming(audio_chunks):
                results.append(text)
            
            assert len(results) == 3
            assert "Hello" in results
            assert "world" in results
            assert "test" in results
    
    def test_is_ready_true(self):
        """Test is_ready when service is initialized."""
        self.service.is_initialized = True
        self.service.fast_model = Mock()  # Add a model so is_ready returns True
        assert self.service.is_ready() is True
    
    def test_is_ready_false(self):
        """Test is_ready when service is not initialized."""
        self.service.is_initialized = False
        assert self.service.is_ready() is False
    
    async def test_cleanup_success(self):
        """Test successful cleanup."""
        self.service.model = Mock()
        self.service.fast_model = Mock()
        self.service.is_initialized = True
        
        await self.service.cleanup()
        
        assert self.service.model is None
        assert self.service.fast_model is None
        assert self.service.is_initialized is False
    
    async def test_cleanup_with_exception(self):
        """Test cleanup with exception handling."""
        self.service.model = Mock()
        self.service.model.cpu.side_effect = Exception("Cleanup failed")
        self.service.is_initialized = True
        
        # Should not raise exception
        await self.service.cleanup()
        
        assert self.service.is_initialized is False
    
    async def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = await self.service.get_supported_languages()
        
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "en" in languages
        assert "es" in languages
    
    async def test_detect_language_success(self):
        """Test language detection."""
        self.service.is_initialized = True
        self.service.model_type = "faster_whisper"
        self.service.fast_model = Mock()
        
        from tests.mocks.ml_dependencies import MockSegment
        
        with patch.object(self.service, '_preprocess_audio') as mock_preprocess:
            
            mock_preprocess.return_value = [0.1, 0.2, 0.3]
            
            # Mock the transcribe method to return Spanish language detection
            mock_segments = [MockSegment(0.0, 2.5, "Hola mundo", 0.95)]
            mock_info = Mock()
            mock_info.language = "es"
            mock_info.language_probability = 0.95
            mock_info.duration = 2.5
            
            self.service.fast_model.transcribe.return_value = (mock_segments, mock_info)
            
            language = await self.service.detect_language(b"fake_audio_data")
            
            assert language == "es"
    
    async def test_detect_language_not_initialized(self):
        """Test language detection when service not initialized."""
        # Ensure service is not initialized
        self.service.is_initialized = False
        self.service.model_type = "faster_whisper"
        self.service.fast_model = None
        
        with pytest.raises(RuntimeError, match="Transcription service not initialized"):
            await self.service.detect_language(b"fake_audio_data") 