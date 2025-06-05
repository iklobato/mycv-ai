"""
Unit tests for Pydantic schemas and validation.

Tests cover:
- Schema validation edge cases
- Custom validators
- Error conditions
- Data type validation
"""

import pytest
from pydantic import ValidationError

from backend.models.schemas import (
    LLMRequest, LLMResponse,
    TranscriptionRequest, TranscriptionResponse,
    TTSRequest, TTSResponse,
    AnimationRequest, AnimationResponse,
    AudioFormat,
    validate_audio_data, validate_image_data
)


class TestLLMRequest:
    """Test LLMRequest schema validation."""
    
    def test_llm_request_valid(self):
        """Test valid LLM request creation."""
        request = LLMRequest(
            message="Hello world",
            conversation_id="test123"
        )
        assert request.message == "Hello world"
        assert request.conversation_id == "test123"
    
    def test_llm_request_whitespace_only_message(self):
        """Test LLM request with whitespace-only message."""
        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(message="   \n\t   ")
        
        error = exc_info.value.errors()[0]
        assert "Message cannot be empty or whitespace only" in str(error)
    
    def test_llm_request_message_stripped(self):
        """Test that message is stripped of whitespace."""
        request = LLMRequest(message="  Hello world  ")
        assert request.message == "Hello world"


class TestValidationFunctions:
    """Test standalone validation functions."""
    
    def test_validate_audio_data_valid(self):
        """Test valid audio data."""
        audio_data = b"fake_audio_data"
        result = validate_audio_data(audio_data)
        assert result == audio_data
    
    def test_validate_audio_data_empty(self):
        """Test empty audio data validation."""
        with pytest.raises(ValueError, match="Audio data cannot be empty"):
            validate_audio_data(b"")
    
    def test_validate_audio_data_too_large(self):
        """Test audio data too large validation."""
        large_audio = b"x" * (51 * 1024 * 1024)  # 51MB
        with pytest.raises(ValueError, match="Audio file too large"):
            validate_audio_data(large_audio)
    
    def test_validate_image_data_valid(self):
        """Test valid image data."""
        image_data = b"fake_image_data"
        result = validate_image_data(image_data)
        assert result == image_data
    
    def test_validate_image_data_empty(self):
        """Test empty image data validation."""
        with pytest.raises(ValueError, match="Image data cannot be empty"):
            validate_image_data(b"")
    
    def test_validate_image_data_too_large(self):
        """Test image data too large validation."""
        large_image = b"x" * (11 * 1024 * 1024)  # 11MB
        with pytest.raises(ValueError, match="Image file too large"):
            validate_image_data(large_image)


class TestSchemaCreation:
    """Test schema creation and basic functionality."""
    
    def test_transcription_request_valid(self):
        """Test valid transcription request creation."""
        audio_data = b"fake_audio_data"
        request = TranscriptionRequest(
            audio_data=audio_data,
            language="en"
        )
        assert request.audio_data == audio_data
        assert request.language == "en"
    
    def test_animation_request_valid_default_avatar(self):
        """Test valid animation request with default avatar."""
        audio_data = b"fake_audio_data"
        request = AnimationRequest(
            audio_data=audio_data,
            use_default_avatar=True
        )
        assert request.audio_data == audio_data
        assert request.use_default_avatar is True
    
    def test_animation_request_valid_custom_image(self):
        """Test valid animation request with custom image."""
        audio_data = b"fake_audio_data"
        image_data = b"fake_image_data"
        request = AnimationRequest(
            audio_data=audio_data,
            image_data=image_data,
            use_default_avatar=False
        )
        assert request.audio_data == audio_data
        assert request.image_data == image_data
        assert request.use_default_avatar is False
    
    def test_tts_request_valid(self):
        """Test valid TTS request creation."""
        request = TTSRequest(
            text="Hello world",
            voice_id="default",
            format=AudioFormat.WAV,
            language="en"
        )
        assert request.text == "Hello world"
        assert request.voice_id == "default"
        assert request.format == AudioFormat.WAV
        assert request.language == "en"
    
    def test_tts_request_defaults(self):
        """Test TTS request with default values."""
        request = TTSRequest(text="Hello world")
        assert request.voice_id == "default"
        assert request.format == AudioFormat.WAV
        assert request.sample_rate == 22050
        assert request.speed == 1.0
        assert request.pitch == 1.0
        assert request.language == "en"


class TestResponses:
    """Test response schemas."""
    
    def test_llm_response_success(self):
        """Test successful LLM response."""
        response = LLMResponse(
            response="Hello there!",
            model="llama3.2",
            success=True,
            processing_time=1.0
        )
        assert response.success is True
        assert response.response == "Hello there!"
    
    def test_transcription_response_success(self):
        """Test successful transcription response."""
        response = TranscriptionResponse(
            text="Hello world",
            language="en",
            confidence=0.95,
            processing_time=1.0,
            word_count=2,
            success=True
        )
        assert response.success is True
        assert response.text == "Hello world"
        assert response.word_count == 2
    
    def test_tts_response_success(self):
        """Test successful TTS response."""
        response = TTSResponse(
            audio_data=b"fake_audio",
            audio_url="http://test.com/audio.wav",
            duration=2.5,
            format=AudioFormat.WAV,
            sample_rate=22050,
            success=True,
            processing_time=1.0
        )
        assert response.success is True
        assert response.audio_data == b"fake_audio"
        assert response.duration == 2.5
    
    def test_animation_response_success(self):
        """Test successful animation response."""
        response = AnimationResponse(
            video_data=b"fake_video",
            video_url="http://test.com/video.mp4",
            duration=3.0,
            fps=25,
            resolution="512x512",
            success=True,
            processing_time=2.0
        )
        assert response.success is True
        assert response.video_data == b"fake_video"
        assert response.fps == 25


class TestAudioFormat:
    """Test AudioFormat enum."""
    
    def test_audio_format_values(self):
        """Test AudioFormat enum values."""
        assert AudioFormat.WAV == "wav"
        assert AudioFormat.MP3 == "mp3"
        assert AudioFormat.OGG == "ogg"
        assert AudioFormat.FLAC == "flac" 