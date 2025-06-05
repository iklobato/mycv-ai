"""
Unit tests for the TTSService.

Tests cover:
- Service initialization and configuration
- Text-to-speech synthesis functionality
- Voice cloning and management
- Error handling and edge cases
- Audio processing and manipulation
- Voice training and streaming synthesis
"""

import pytest
import tempfile
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, mock_open, PropertyMock
from pathlib import Path
import numpy as np

# Import the service (mocks are set up in conftest.py)
from backend.services.tts_service import TTSService
from backend.models.schemas import TTSRequest, TTSResponse


class TestTTSService:
    """Test suite for TTSService."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.service = TTSService()
    
    def teardown_method(self):
        """Clean up after each test."""
        pass
    
    def test_tts_service_initialization(self):
        """Test TTSService initialization."""
        service = TTSService()
        
        assert service.tts_model is None
        assert service.device is None
        assert service.is_initialized is False
        assert service.voice_samples == {}
        assert service.default_speaker == "default"
        assert len(service.supported_languages) > 0
        assert "en" in service.supported_languages
    
    async def test_initialize_success(self):
        """Test successful service initialization."""
        with patch('backend.services.tts_service.settings') as mock_settings:
            mock_settings.TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
            mock_settings.USE_GPU = False
            mock_settings.VOICE_SAMPLES_DIR = Path("/tmp/voice_samples")
            
            with patch.object(self.service, '_load_xtts_model') as mock_load_model, \
                 patch.object(self.service, '_load_voice_samples') as mock_load_voices:
                
                await self.service.initialize()
                
                assert self.service.is_initialized is True
                assert self.service.device == "cpu"
                mock_load_model.assert_called_once()
                mock_load_voices.assert_called_once()
    
    async def test_initialize_failure(self):
        """Test service initialization failure."""
        with patch('backend.services.tts_service.settings') as mock_settings:
            mock_settings.TTS_MODEL = "invalid_model"
            mock_settings.USE_GPU = False
            mock_settings.VOICE_SAMPLES_DIR = Path("/tmp/voice_samples")
            
            with patch.object(self.service, '_load_xtts_model', side_effect=Exception("Model load failed")):
                with pytest.raises(Exception, match="Model load failed"):
                    await self.service.initialize()
    
    def test_get_device_cpu(self):
        """Test device selection (CPU)."""
        device = self.service._get_device()
        assert device == "cpu"
    
    def test_get_device_cuda(self):
        """Test device selection (CUDA)."""
        with patch('backend.services.tts_service.settings') as mock_settings, \
             patch('backend.services.tts_service.torch') as mock_torch:
            mock_settings.USE_GPU = True
            mock_torch.cuda.is_available.return_value = True
            
            device = self.service._get_device()
            assert device == "cuda"
    
    def test_get_device_mps(self):
        """Test device selection (MPS)."""
        with patch('backend.services.tts_service.settings') as mock_settings, \
             patch('backend.services.tts_service.torch') as mock_torch:
            mock_settings.USE_GPU = True
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = True
            
            device = self.service._get_device()
            assert device == "mps"
    
    async def test_load_xtts_model_success(self):
        """Test successful XTTS model loading."""
        with patch('backend.services.tts_service.settings') as mock_settings:
            mock_settings.TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
            
            with patch('backend.services.tts_service.TTS') as mock_tts_class:
                mock_model = Mock()
                mock_model.is_multi_speaker = True
                mock_model.is_multi_lingual = True
                mock_tts_class.return_value = mock_model
                
                await self.service._load_xtts_model()
                
                assert self.service.tts_model == mock_model
                mock_tts_class.assert_called_once()
    
    async def test_load_xtts_model_fallback(self):
        """Test XTTS model loading with fallback."""
        with patch('backend.services.tts_service.settings') as mock_settings:
            mock_settings.TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
            
            with patch('backend.services.tts_service.TTS') as mock_tts_class:
                # First call fails, second call succeeds (fallback)
                mock_tts_class.side_effect = [
                    Exception("XTTS failed"),
                    Mock()
                ]
                
                await self.service._load_xtts_model()
                
                assert mock_tts_class.call_count == 2
    
    async def test_load_xtts_model_fallback_complete_failure(self):
        """Test XTTS model loading with complete fallback failure."""
        with patch('backend.services.tts_service.settings') as mock_settings:
            mock_settings.TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
            
            with patch('backend.services.tts_service.TTS') as mock_tts_class:
                # Both calls fail (original and fallback)
                mock_tts_class.side_effect = [
                    Exception("XTTS failed"),
                    Exception("Fallback TTS failed")
                ]
                
                with pytest.raises(Exception, match="Fallback TTS failed"):
                    await self.service._load_xtts_model()
    
    async def test_load_voice_samples_with_files(self):
        """Test loading voice samples when files exist."""
        with patch('backend.services.tts_service.settings') as mock_settings:
            voice_dir = Path("/tmp/voice_samples")
            mock_settings.VOICE_SAMPLES_DIR = voice_dir
            
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.glob') as mock_glob:
                
                # Mock finding voice files
                mock_files = [Path("/tmp/voice_samples/voice1.wav"), Path("/tmp/voice_samples/voice2.mp3")]
                mock_glob.return_value = mock_files
                
                await self.service._load_voice_samples()
                
                assert self.service.voice_samples[self.service.default_speaker] == str(mock_files[0])
    
    async def test_load_voice_samples_no_directory(self):
        """Test loading voice samples when directory doesn't exist."""
        with patch('backend.services.tts_service.settings') as mock_settings:
            voice_dir = Path("/tmp/nonexistent")
            mock_settings.VOICE_SAMPLES_DIR = voice_dir
            
            with patch('pathlib.Path.exists', return_value=False):
                await self.service._load_voice_samples()
                
                # Should complete without error
                assert len(self.service.voice_samples) == 0
    
    async def test_load_voice_samples_no_files_found(self):
        """Test loading voice samples when directory exists but no files found."""
        with patch('backend.services.tts_service.settings') as mock_settings:
            voice_dir = Path("/tmp/voice_samples")
            mock_settings.VOICE_SAMPLES_DIR = voice_dir
            
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.glob', return_value=[]):  # No files found
                
                await self.service._load_voice_samples()
                
                # Should log info message about no files found
                assert len(self.service.voice_samples) == 0
    
    async def test_synthesize_success(self):
        """Test successful speech synthesis."""
        self.service.is_initialized = True
        self.service.tts_model = Mock()
        
        with patch.object(self.service, '_clean_text', return_value="Hello world") as mock_clean, \
             patch.object(self.service, '_generate_speech', return_value=b"fake_audio") as mock_generate, \
             patch.object(self.service, '_save_audio_temp', return_value="http://test.com/audio.wav") as mock_save:
            
            request = TTSRequest(
                text="Hello world",
                voice_id="default",
                format="wav",
                sample_rate=22050,
                language="en"
            )
            
            response = await self.service.synthesize(request)
            
            assert response.success is True
            assert response.audio_data == b"fake_audio"
            assert response.audio_url == "http://test.com/audio.wav"
            assert response.processing_time > 0
            
            mock_clean.assert_called_once_with("Hello world")
            mock_generate.assert_called_once()
            mock_save.assert_called_once()
    
    async def test_synthesize_not_initialized(self):
        """Test synthesis when service not initialized."""
        request = TTSRequest(
            text="Hello world",
            voice_id="default",
            format="wav",
            language="en"
        )
        
        with pytest.raises(RuntimeError, match="TTS service not initialized"):
            await self.service.synthesize(request)
    
    async def test_synthesize_empty_text(self):
        """Test synthesis with empty text."""
        self.service.is_initialized = True
        
        # Create a valid request first
        request = TTSRequest(
            text="valid text",  # Valid text to pass schema validation
            voice_id="default",
            format="wav",
            language="en"
        )
        
        # Mock _clean_text to return empty string after validation
        with patch.object(self.service, '_clean_text', return_value=""):
            response = await self.service.synthesize(request)
            
            assert response.success is False
            assert "Empty text provided" in response.error_message
    
    async def test_synthesize_exception_handling(self):
        """Test synthesis exception handling."""
        self.service.is_initialized = True
        
        with patch.object(self.service, '_clean_text', side_effect=Exception("Text cleaning failed")):
            request = TTSRequest(
                text="Hello world",
                voice_id="default",
                format="wav",
                language="en"
            )
            
            response = await self.service.synthesize(request)
            
            assert response.success is False
            assert "Text cleaning failed" in response.error_message
    
    async def test_speech_generation_failure(self):
        """Test speech generation failure handling."""
        self.service.is_initialized = True
        self.service.tts_model = Mock()
        
        with patch.object(self.service, '_clean_text', return_value="Hello world"), \
             patch.object(self.service, '_generate_speech', side_effect=Exception("Speech generation failed")):
            
            request = TTSRequest(
                text="Hello world",
                voice_id="default",
                format="wav",
                language="en"
            )
            
            response = await self.service.synthesize(request)
            
            assert response.success is False
            assert "Speech generation failed" in response.error_message
    
    def test_clean_text_success(self):
        """Test text cleaning."""
        dirty_text = "  Hello... world!!!   "
        cleaned = self.service._clean_text(dirty_text)
        
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
    
    async def test_generate_speech_with_cloned_voice(self):
        """Test speech generation with cloned voice."""
        self.service.tts_model = Mock()
        self.service.voice_samples = {"custom_voice": "/path/to/voice.wav"}
        
        with patch.object(self.service, '_generate_cloned_speech', return_value=b"cloned_audio") as mock_cloned:
            request = TTSRequest(
                text="Hello world",
                voice_id="custom_voice",
                language="en",
                format="wav"
            )
            
            result = await self.service._generate_speech(request, "Hello world")
            
            assert result == b"cloned_audio"
            mock_cloned.assert_called_once()
    
    async def test_generate_speech_standard(self):
        """Test standard speech generation."""
        self.service.tts_model = Mock()
        
        with patch.object(self.service, '_generate_standard_speech', return_value=b"standard_audio") as mock_standard:
            request = TTSRequest(
                text="Hello world",
                voice_id="default",
                language="en",
                format="wav"
            )
            
            result = await self.service._generate_speech(request, "Hello world")
            
            assert result == b"standard_audio"
            mock_standard.assert_called_once()
    
    async def test_generate_cloned_speech_success(self):
        """Test successful voice cloning speech generation."""
        self.service.tts_model = Mock()
        self.service.voice_samples = {"custom_voice": "/path/to/voice.wav"}
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            
            # Mock temporary file context manager
            mock_temp_instance = Mock()
            mock_temp_instance.name = "/tmp/output.wav"
            mock_temp_file.return_value.__enter__.return_value = mock_temp_instance
            
            # Mock TTS model inference
            self.service.tts_model.tts_to_file = Mock()
            
            with patch.object(self.service, '_read_audio_file', return_value=b"output_audio_data") as mock_read:
                
                request = TTSRequest(
                    text="Hello world",
                    voice_id="custom_voice",
                    language="en"
                )
                
                result = await self.service._generate_cloned_speech(request, "Hello world")
                
                assert result == b"output_audio_data"
                assert mock_read.call_count == 1  # Only reads the generated output file
                self.service.tts_model.tts_to_file.assert_called_once()
    
    async def test_generate_cloned_speech_voice_not_found(self):
        """Test voice cloning with voice not found error."""
        self.service.tts_model = Mock()
        self.service.voice_samples = {"other_voice": "/path/to/other.wav"}
        
        request = TTSRequest(
            text="Hello world",
            voice_id="nonexistent_voice",  # Voice not in samples
            language="en"
        )
        
        with pytest.raises(ValueError, match="No voice sample found for voice_id: nonexistent_voice"):
            await self.service._generate_cloned_speech(request, "Hello world")
    
    async def test_generate_cloned_speech_exception(self):
        """Test voice cloning with exception handling."""
        self.service.tts_model = Mock()
        self.service.voice_samples = {"custom_voice": "/path/to/voice.wav"}
        
        with patch.object(self.service, '_read_audio_file', side_effect=Exception("Voice cloning failed")):
            request = TTSRequest(
                text="Hello world",
                voice_id="custom_voice",
                language="en"
            )
            
            with pytest.raises(Exception, match="Voice cloning failed"):
                await self.service._generate_cloned_speech(request, "Hello world")
    
    async def test_generate_standard_speech_success(self):
        """Test standard speech generation."""
        self.service.tts_model = Mock()
        self.service.default_speaker = "default"
        
        # Mock TTS model methods
        self.service.tts_model.tts_to_file = Mock()
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
             patch.object(self.service, '_read_audio_file', return_value=b"audio_data") as mock_read, \
             patch.object(self.service, '_modify_audio', return_value=b"modified_audio_data") as mock_modify:
            
            # Mock temporary file context manager
            mock_temp_instance = Mock()
            mock_temp_instance.name = "/tmp/output.wav"
            mock_temp_file.return_value.__enter__.return_value = mock_temp_instance
            
            request = TTSRequest(
                text="Hello world",
                voice_id="default",
                language="en",
                format="wav",
                speed=1.2,
                pitch=1.1
            )
            
            result = await self.service._generate_standard_speech(request, "Hello world")
            
            assert result == b"modified_audio_data"
            self.service.tts_model.tts_to_file.assert_called_once()
            mock_read.assert_called_once()
            mock_modify.assert_called_once()
    
    async def test_generate_standard_speech_exception(self):
        """Test standard speech generation with exception handling."""
        self.service.tts_model = Mock()
        self.service.default_speaker = "default"
        
        # Mock TTS model to raise exception
        self.service.tts_model.tts_to_file.side_effect = Exception("Standard TTS failed")
        
        request = TTSRequest(
            text="Hello world",
            voice_id="default",
            language="en"
        )
        
        with pytest.raises(Exception, match="Standard TTS failed"):
            await self.service._generate_standard_speech(request, "Hello world")
    
    async def test_read_audio_file_success(self):
        """Test successful audio file reading."""
        file_path = "/path/to/audio.wav"
        
        # Mock stereo audio data
        stereo_audio = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        
        with patch('backend.services.tts_service.sf.read', return_value=(stereo_audio, 22050)), \
             patch('backend.services.tts_service.sf.write') as mock_write, \
             patch('backend.services.tts_service.io.BytesIO') as mock_bytes_io:
            
            # Mock BytesIO
            mock_buffer = Mock()
            mock_buffer.getvalue.return_value = b"converted_audio_data"
            mock_bytes_io.return_value = mock_buffer
            
            result = await self.service._read_audio_file(file_path)
            
            assert result == b"converted_audio_data"
            mock_write.assert_called_once()
    
    async def test_read_audio_file_mono_success(self):
        """Test successful audio file reading with mono audio."""
        file_path = "/path/to/audio.wav"
        
        # Mock mono audio data
        mono_audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        with patch('backend.services.tts_service.sf.read', return_value=(mono_audio, 22050)), \
             patch('backend.services.tts_service.sf.write') as mock_write, \
             patch('backend.services.tts_service.io.BytesIO') as mock_bytes_io:
            
            # Mock BytesIO
            mock_buffer = Mock()
            mock_buffer.getvalue.return_value = b"mono_audio_data"
            mock_bytes_io.return_value = mock_buffer
            
            result = await self.service._read_audio_file(file_path)
            
            assert result == b"mono_audio_data"
            mock_write.assert_called_once()
    
    async def test_read_audio_file_exception(self):
        """Test audio file reading with exception handling."""
        file_path = "/path/to/audio.wav"
        
        with patch('backend.services.tts_service.sf.read', side_effect=Exception("Failed to read audio file")):
            with pytest.raises(Exception, match="Failed to read audio file"):
                await self.service._read_audio_file(file_path)
    
    def test_adjust_pitch_success(self):
        """Test pitch adjustment."""
        audio_data = b"fake_audio_data"
        sample_rate = 22050
        pitch_shift = 1.2
        
        with patch('backend.services.tts_service.AudioSegment.from_wav') as mock_from_wav, \
             patch('backend.services.tts_service.io.BytesIO') as mock_bytes_io:
            
            # Mock audio segment
            mock_segment = Mock()
            mock_segment.raw_data = b"raw_audio"
            mock_segment._spawn.return_value = mock_segment
            mock_segment.set_frame_rate.return_value = mock_segment
            mock_segment.export.return_value = None
            mock_from_wav.return_value = mock_segment
            
            # Mock BytesIO for output
            mock_output = Mock()
            mock_output.getvalue.return_value = b"pitch_adjusted_audio"
            mock_bytes_io.return_value = mock_output
            
            result = self.service._adjust_pitch(audio_data, sample_rate, pitch_shift)
            
            assert result == b"pitch_adjusted_audio"
    
    def test_adjust_pitch_failure(self):
        """Test pitch adjustment with error handling."""
        audio_data = b"fake_audio_data"
        sample_rate = 22050
        pitch_shift = 1.2
        
        with patch('backend.services.tts_service.AudioSegment.from_wav', side_effect=Exception("Pitch adjustment failed")):
            result = self.service._adjust_pitch(audio_data, sample_rate, pitch_shift)
            
            # Should return original audio on failure
            assert result == audio_data
    
    def test_modify_audio_success(self):
        """Test audio modification."""
        audio_data = b"fake_audio_data"
        sample_rate = 22050
        speed = 1.2
        pitch = 1.1
        
        with patch('backend.services.tts_service.AudioSegment.from_wav') as mock_from_wav, \
             patch('backend.services.tts_service.io.BytesIO') as mock_bytes_io:
            
            # Mock audio segment
            mock_segment = Mock()
            mock_segment.speedup.return_value = mock_segment
            mock_segment.raw_data = b"raw_audio"
            mock_segment._spawn.return_value = mock_segment
            mock_segment.set_frame_rate.return_value = mock_segment
            mock_segment.export.return_value = None
            mock_from_wav.return_value = mock_segment
            
            # Mock BytesIO for output
            mock_output = Mock()
            mock_output.getvalue.return_value = b"modified_audio"
            mock_bytes_io.return_value = mock_output
            
            result = self.service._modify_audio(audio_data, sample_rate, speed, pitch)
            
            assert result == b"modified_audio"
    
    def test_modify_audio_failure(self):
        """Test audio modification with error handling."""
        audio_data = b"fake_audio_data"
        sample_rate = 22050
        speed = 1.2
        pitch = 1.1
        
        with patch('backend.services.tts_service.AudioSegment.from_wav', side_effect=Exception("Audio modification failed")):
            result = self.service._modify_audio(audio_data, sample_rate, speed, pitch)
            
            # Should return original audio on failure
            assert result == audio_data
    
    async def test_save_audio_temp_success(self):
        """Test saving audio to temporary file."""
        audio_data = b"fake_audio_data"
        file_format = "wav"
        
        with patch('backend.services.tts_service.settings') as mock_settings:
            temp_dir = Path("/tmp")
            mock_settings.TEMP_DIR = temp_dir
            
            result = await self.service._save_audio_temp(audio_data, file_format)
            
            # Check that result contains the expected pattern
            assert result.startswith("/static/temp/tts_")
            assert result.endswith(".wav")
    
    async def test_save_audio_temp_failure(self):
        """Test saving audio to temporary file with error handling."""
        audio_data = b"fake_audio_data"
        file_format = "wav"
        
        with patch('builtins.open', side_effect=Exception("Failed to save temporary audio file")):
            result = await self.service._save_audio_temp(audio_data, file_format)
            
            # Should return None on failure
            assert result is None
    
    async def test_train_voice_success(self):
        """Test voice training."""
        voice_samples = [b"sample1", b"sample2"]
        speaker_name = "new_voice"
        
        with patch('backend.services.tts_service.settings') as mock_settings:
            mock_settings.TEMP_DIR = Path("/tmp")
            
            result = await self.service.train_voice(voice_samples, speaker_name)
            
            # Check successful response
            assert result["status"] == "ready"
            assert result["voice_id"] == speaker_name
    
    async def test_train_voice_no_samples(self):
        """Test voice training with no samples provided."""
        voice_samples = []  # Empty samples list
        speaker_name = "new_voice"
        
        result = await self.service.train_voice(voice_samples, speaker_name)
        
        # Should return error status
        assert result["status"] == "error"
        assert "No voice samples provided" in result["error"]
    
    async def test_train_voice_exception(self):
        """Test voice training with exception handling."""
        voice_samples = [b"sample1", b"sample2"]
        speaker_name = "new_voice"
        
        # Mock file writing to fail
        with patch('builtins.open', side_effect=Exception("Voice training failed")):
            result = await self.service.train_voice(voice_samples, speaker_name)
            
            # Should return error status
            assert result["status"] == "error"
            assert "Voice training failed" in result["error"]
    
    async def test_get_voices_success(self):
        """Test getting available voices."""
        self.service.voice_samples = {
            "default": "/path/to/default.wav",
            "custom": "/path/to/custom.wav"
        }
        
        result = await self.service.get_voices()
        
        assert len(result) >= 1  # At least the default voice
        # Check that all voices have the required voice_id field
        for voice in result:
            assert "voice_id" in voice
    
    async def test_synthesize_streaming_success(self):
        """Test streaming synthesis."""
        self.service.is_initialized = True
        self.service.tts_model = Mock()
        
        text = "Hello world from streaming"
        
        with patch.object(self.service, '_split_into_sentences', return_value=["Hello", "world", "from", "streaming"]), \
             patch.object(self.service, '_generate_speech', return_value=b"audio_chunk") as mock_generate:
            
            results = []
            
            async for chunk in self.service.synthesize_streaming(text, "default"):
                results.append(chunk)
            
            assert len(results) == 4  # Should have 4 chunks for 4 sentences
            assert all(chunk == b"audio_chunk" for chunk in results)
            assert mock_generate.call_count == 4
    
    async def test_synthesize_streaming_exception(self):
        """Test streaming synthesis with exception handling."""
        self.service.is_initialized = True
        self.service.tts_model = Mock()
        
        text = "Hello world"
        
        with patch.object(self.service, '_split_into_sentences', return_value=["Hello", "world"]), \
             patch.object(self.service, '_generate_speech', side_effect=Exception("Streaming synthesis failed")):
            
            results = []
            
            async for chunk in self.service.synthesize_streaming(text, "default"):
                results.append(chunk)
            
            # Should handle exceptions gracefully and continue or stop
            assert len(results) == 0  # No successful chunks due to exception
    
    def test_split_into_sentences_success(self):
        """Test sentence splitting."""
        text = "Hello world. How are you? I am fine!"
        
        sentences = self.service._split_into_sentences(text)
        
        assert len(sentences) >= 2
        assert any("Hello world" in sentence for sentence in sentences)
    
    def test_is_ready_true(self):
        """Test is_ready when service is initialized."""
        self.service.is_initialized = True
        self.service.tts_model = Mock()  # Add a model so is_ready returns True
        assert self.service.is_ready() is True
    
    def test_is_ready_false(self):
        """Test is_ready when service is not initialized."""
        self.service.is_initialized = False
        assert self.service.is_ready() is False
    
    async def test_cleanup_success(self):
        """Test successful cleanup."""
        self.service.tts_model = Mock()
        self.service.is_initialized = True
        
        await self.service.cleanup()
        
        assert self.service.tts_model is None
        assert self.service.is_initialized is False
    
    async def test_cleanup_with_exception(self):
        """Test cleanup with exception handling."""
        self.service.tts_model = Mock()
        self.service.is_initialized = True
        
        # Mock torch.cuda.empty_cache to raise an exception
        with patch('backend.services.tts_service.torch') as mock_torch:
            mock_torch.cuda.empty_cache.side_effect = Exception("Error during TTS service cleanup")
            
            # Should not raise exception despite cleanup error
            await self.service.cleanup()
            
            assert self.service.tts_model is None
            assert self.service.is_initialized is False
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = self.service.get_supported_languages()
        
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "en" in languages
        assert "es" in languages
    
    async def test_get_model_info_success(self):
        """Test getting model information."""
        self.service.tts_model = Mock()
        self.service.is_initialized = True
        
        # Mock the model name to match test expectations  
        with patch('backend.services.tts_service.settings') as mock_settings:
            mock_settings.TTS_MODEL = "test_model"
            
            result = await self.service.get_model_info()
            
            # Check that result is a dictionary and has some expected fields
            assert isinstance(result, dict)
            assert "initialized" in result 
    
    async def test_get_model_info_not_initialized(self):
        """Test getting model info when service not initialized."""
        self.service.is_initialized = False
        
        result = await self.service.get_model_info()
        
        assert "error" in result
        assert result["error"] == "Service not initialized"
    
    async def test_get_model_info_settings_exception(self):
        """Test get_model_info exception handling when accessing settings fails."""
        self.service.is_initialized = True
        self.service.tts_model = Mock()
        
        # Make accessing settings.TTS_MODEL fail by using PropertyMock
        with patch('backend.services.tts_service.settings') as mock_settings:
            # Configure the mock to raise an exception when TTS_MODEL is accessed
            type(mock_settings).TTS_MODEL = PropertyMock(side_effect=Exception("Settings access failed"))
            
            result = await self.service.get_model_info()
            
            # Should return error dict (lines 608-610)
            assert "error" in result
            assert "Settings access failed" in result["error"] or "Failed to get model info" in result["error"]
    
    async def test_synthesize_streaming_direct_call(self):
        """Test streaming synthesis calling _generate_speech directly (lines 543-544)."""
        self.service.is_initialized = True
        self.service.tts_model = Mock()
        
        # Mock to ensure we go through the streaming path properly
        with patch.object(self.service, '_split_into_sentences', return_value=["Hello.", "World."]), \
             patch.object(self.service, 'synthesize') as mock_synthesize:
            
            # Mock successful synthesize responses
            mock_response = Mock()
            mock_response.success = True
            mock_response.audio_data = b"audio_chunk"
            mock_synthesize.return_value = mock_response
            
            results = []
            async for chunk in self.service.synthesize_streaming("Hello world", "default"):
                results.append(chunk)
            
            # Should process both sentences and yield audio
            assert len(results) == 2
            assert all(chunk == b"audio_chunk" for chunk in results)
            assert mock_synthesize.call_count == 2
    
    async def test_cleanup_success_logging(self):
        """Test successful cleanup with logging (lines 584-585)."""
        self.service.tts_model = Mock()
        self.service.is_initialized = True
        self.service.device = "cpu"
        
        with patch('backend.services.tts_service.logger') as mock_logger:
            await self.service.cleanup()
            
            # Should log successful cleanup and set initialized to False
            mock_logger.info.assert_any_call("TTS service cleaned up successfully")
            assert self.service.is_initialized is False
    
    async def test_cleanup_with_cuda_cache_clearing(self):
        """Test cleanup with CUDA cache clearing (lines 579)."""
        self.service.tts_model = Mock()
        self.service.is_initialized = True
        self.service.device = "cuda"
        
        with patch('backend.services.tts_service.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.empty_cache = Mock()
            
            await self.service.cleanup()
            
            # Should clear CUDA cache (line 579)
            mock_torch.cuda.empty_cache.assert_called_once()
            assert self.service.is_initialized is False
    
    async def test_speech_generation_general_exception(self):
        """Test general exception handling in _generate_speech (lines 253-255)."""
        self.service.tts_model = Mock()
        self.service.voice_samples = {"custom_voice": "/path/to/voice.wav"}
        
        # Mock _generate_cloned_speech to raise a general exception
        with patch.object(self.service, '_generate_cloned_speech', side_effect=Exception("General speech generation error")):
            request = TTSRequest(
                text="Hello world",
                voice_id="custom_voice",
                language="en"
            )
            
            # Should re-raise the exception after logging (lines 253-255)
            with pytest.raises(Exception, match="General speech generation error"):
                await self.service._generate_speech(request, "Hello world")
    
    async def test_cleanup_with_exception_in_cleanup(self):
        """Test cleanup with exception during cleanup operations (lines 584-585)."""
        self.service.tts_model = Mock()
        self.service.is_initialized = True
        
        # Mock voice_samples.clear() to raise an exception
        self.service.voice_samples = Mock()
        self.service.voice_samples.clear.side_effect = Exception("Cleanup error")
        
        with patch('backend.services.tts_service.logger') as mock_logger:
            # Should not raise exception despite cleanup error
            await self.service.cleanup()
            
            # Should log the error (line 585)
            mock_logger.error.assert_called_with("Error during TTS service cleanup: Cleanup error")
            # Should still set attributes to None despite exception
            assert self.service.tts_model is None 
    
    async def test_synthesize_streaming_success_path(self):
        """Test streaming synthesis with successful audio generation (lines 540-541)."""
        self.service.is_initialized = True
        self.service.tts_model = Mock()
        
        with patch.object(self.service, '_split_into_sentences', return_value=["Hello.", "World."]):
            
            # Mock successful synthesize that actually creates TTSResponse objects
            async def mock_synthesize(request):
                response = TTSResponse(
                    success=True,
                    audio_data=b"audio_chunk",
                    audio_url="http://test.com/audio.wav",
                    processing_time=0.1,
                    format="wav",
                    sample_rate=22050,
                    duration=1.0  # Add missing duration field
                )
                return response
            
            with patch.object(self.service, 'synthesize', side_effect=mock_synthesize):
                results = []
                async for chunk in self.service.synthesize_streaming("Hello world", "default"):
                    results.append(chunk)
                
                # Should yield audio data for each successful synthesis
                assert len(results) == 2
                assert all(chunk == b"audio_chunk" for chunk in results)
    
    async def test_synthesize_streaming_exception_path(self):
        """Test streaming synthesis exception handling (line 544)."""
        self.service.is_initialized = True
        self.service.tts_model = Mock()
        
        with patch.object(self.service, '_split_into_sentences', side_effect=Exception("Sentence splitting failed")), \
             patch('backend.services.tts_service.logger') as mock_logger:
            
            results = []
            async for chunk in self.service.synthesize_streaming("Hello world", "default"):
                results.append(chunk)
            
            # Should handle exception and log error
            mock_logger.error.assert_called_with("Streaming synthesis failed: Sentence splitting failed")
            assert len(results) == 0  # No results due to exception 
    
    async def test_generate_cloned_speech_with_pitch_adjustment(self):
        """Test voice cloning with pitch adjustment to cover line 284."""
        request = TTSRequest(
            text="Test speech",
            voice_id="custom_voice",
            speed=1.0,
            pitch=1.5,  # This will trigger pitch adjustment (line 284)
            sample_rate=22050
        )
        
        # Mock service setup
        self.service.is_initialized = True
        self.service.device = "cpu"
        
        with patch('backend.services.tts_service.tempfile.NamedTemporaryFile') as mock_temp, \
             patch('backend.services.tts_service.TTS') as mock_tts, \
             patch.object(self.service, '_read_audio_file', return_value=b"audio_data"), \
             patch.object(self.service, '_adjust_pitch', return_value=b"pitched_audio") as mock_adjust_pitch, \
             patch('backend.services.tts_service.Path') as mock_path:
            
            # Setup temp file mock
            mock_temp_instance = Mock()
            mock_temp_instance.name = "/tmp/test.wav"
            mock_temp.return_value.__enter__.return_value = mock_temp_instance
            
            # Setup TTS model mock
            mock_tts_instance = Mock()
            mock_tts.return_value = mock_tts_instance
            self.service.tts_model = mock_tts_instance
            self.service.voice_samples = {"custom_voice": "/path/to/sample.wav"}
            
            # Setup path mock
            mock_path_instance = Mock()
            mock_path.return_value = mock_path_instance
            
            result = await self.service._generate_cloned_speech(request, "Test speech")
            
            # Verify pitch adjustment was called (line 284)
            mock_adjust_pitch.assert_called_once_with(b"audio_data", 22050, 1.5)
            assert result == b"pitched_audio"

    async def test_generate_standard_speech_single_language_model(self):
        """Test standard speech generation with single language model to cover line 314."""
        request = TTSRequest(
            text="Test speech",
            voice_id="default",
            language="en",
            speed=1.0,
            pitch=1.0,
            sample_rate=22050
        )
        
        # Mock service setup
        self.service.is_initialized = True
        self.service.device = "cpu"
        
        with patch('backend.services.tts_service.tempfile.NamedTemporaryFile') as mock_temp, \
             patch.object(self.service, '_read_audio_file', return_value=b"audio_data"), \
             patch('backend.services.tts_service.Path') as mock_path:
            
            # Setup temp file mock
            mock_temp_instance = Mock()
            mock_temp_instance.name = "/tmp/test.wav"
            mock_temp.return_value.__enter__.return_value = mock_temp_instance
            
            # Setup TTS model mock (single language model)
            mock_tts_model = Mock()
            mock_tts_model.is_multi_lingual = False  # This will trigger line 314
            self.service.tts_model = mock_tts_model
            
            # Setup path mock
            mock_path_instance = Mock()
            mock_path.return_value = mock_path_instance
            
            result = await self.service._generate_standard_speech(request, "Test speech")
            
            # Verify single language model path was taken (line 314)
            mock_tts_model.tts_to_file.assert_called_once_with(
                text="Test speech",
                file_path="/tmp/test.wav"
            )
            assert result == b"audio_data" 