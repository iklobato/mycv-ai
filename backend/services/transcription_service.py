"""
Transcription Service using OpenAI Whisper

Handles speech-to-text conversion with support for:
- Multiple audio formats
- Language detection and specification
- Real-time and batch processing
- GPU acceleration
"""

import asyncio
import logging
import time
import tempfile
import io
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import whisper
import librosa
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

from ..models.schemas import TranscriptionRequest, TranscriptionResponse
from ..config import settings

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Service for converting audio to text using Whisper."""
    
    def __init__(self):
        self.model = None
        self.fast_model = None
        self.device = None
        self.is_initialized = False
        self.model_type = "faster_whisper"  # or "openai_whisper"
        
    async def initialize(self):
        """Initialize the Whisper model."""
        logger.info("Initializing Whisper transcription service...")
        
        try:
            # Determine device
            self.device = self._get_device()
            logger.info(f"Using device: {self.device}")
            
            # Load model based on configuration
            if self.model_type == "faster_whisper":
                await self._load_faster_whisper_model()
            else:
                await self._load_openai_whisper_model()
            
            self.is_initialized = True
            logger.info("Whisper transcription service initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize transcription service: {e}")
            raise
    
    async def _load_faster_whisper_model(self):
        """Load Faster Whisper model for better performance."""
        try:
            model_config = settings.MODELS_CONFIG["whisper"]
            
            self.fast_model = WhisperModel(
                model_size_or_path=settings.WHISPER_MODEL,
                device=self.device,
                compute_type=model_config["compute_type"],
                num_workers=model_config["num_workers"]
            )
            
            logger.info(f"Loaded Faster Whisper model: {settings.WHISPER_MODEL}")
            
        except Exception as e:
            logger.error(f"Failed to load Faster Whisper model: {e}")
            # Fallback to OpenAI Whisper
            await self._load_openai_whisper_model()
    
    async def _load_openai_whisper_model(self):
        """Load OpenAI Whisper model as fallback."""
        try:
            self.model = whisper.load_model(
                settings.WHISPER_MODEL,
                device=self.device
            )
            self.model_type = "openai_whisper"
            
            logger.info(f"Loaded OpenAI Whisper model: {settings.WHISPER_MODEL}")
            
        except Exception as e:
            logger.error(f"Failed to load OpenAI Whisper model: {e}")
            raise
    
    def _get_device(self) -> str:
        """Determine the best device to use."""
        if settings.WHISPER_DEVICE == "auto":
            if torch.cuda.is_available() and settings.USE_GPU:
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return settings.WHISPER_DEVICE
    
    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """
        Transcribe audio to text.
        
        Args:
            request: Transcription request with audio data
            
        Returns:
            TranscriptionResponse with transcribed text
        """
        if not self.is_initialized:
            raise RuntimeError("Transcription service not initialized")
        
        start_time = time.time()
        
        try:
            # Preprocess audio
            audio_array = await self._preprocess_audio(request.audio_data)
            
            # Transcribe based on model type
            if self.model_type == "faster_whisper" and self.fast_model:
                result = await self._transcribe_faster_whisper(audio_array, request)
            else:
                result = await self._transcribe_openai_whisper(audio_array, request)
            
            processing_time = time.time() - start_time
            
            # Create response
            response = TranscriptionResponse(
                text=result["text"],
                confidence=result.get("confidence", 0.0),
                language=result.get("language", request.language),
                duration=result.get("duration", 0.0),
                words=result.get("words"),
                processing_time=processing_time
            )
            
            logger.debug(f"Transcription completed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            processing_time = time.time() - start_time
            
            return TranscriptionResponse(
                text="",
                confidence=0.0,
                language=request.language,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _preprocess_audio(self, audio_data: bytes) -> np.ndarray:
        """
        Preprocess audio data for transcription.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Create temporary file for audio processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # Load audio with librosa
                audio_array, sample_rate = librosa.load(
                    temp_path,
                    sr=settings.AUDIO_SAMPLE_RATE,
                    mono=True
                )
                
                # Normalize audio
                audio_array = librosa.util.normalize(audio_array)
                
                # Remove silence at the beginning and end
                audio_array, _ = librosa.effects.trim(
                    audio_array,
                    top_db=20,
                    frame_length=2048,
                    hop_length=512
                )
                
                return audio_array
                
            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            # Try to load directly as numpy array
            audio_buffer = io.BytesIO(audio_data)
            try:
                audio_array, _ = sf.read(audio_buffer)
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)  # Convert to mono
                return audio_array
            except Exception as inner_e:
                logger.error(f"Direct audio loading also failed: {inner_e}")
                raise e
    
    async def _transcribe_faster_whisper(
        self, 
        audio_array: np.ndarray, 
        request: TranscriptionRequest
    ) -> Dict[str, Any]:
        """Transcribe using Faster Whisper model."""
        try:
            # Determine language
            language = None if request.language == "auto" else request.language
            
            # Transcribe
            segments, info = self.fast_model.transcribe(
                audio_array,
                language=language,
                temperature=settings.WHISPER_TEMPERATURE,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Combine segments
            text_parts = []
            words_list = []
            total_duration = 0.0
            
            for segment in segments:
                text_parts.append(segment.text)
                total_duration = max(total_duration, segment.end)
                
                # Add word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        words_list.append({
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        })
            
            full_text = " ".join(text_parts).strip()
            
            # Calculate average confidence
            confidence = info.language_probability if hasattr(info, 'language_probability') else 0.8
            
            return {
                "text": full_text,
                "confidence": confidence,
                "language": info.language if hasattr(info, 'language') else "en",
                "duration": total_duration,
                "words": words_list if words_list else None
            }
            
        except Exception as e:
            logger.error(f"Faster Whisper transcription failed: {e}")
            raise
    
    async def _transcribe_openai_whisper(
        self, 
        audio_array: np.ndarray, 
        request: TranscriptionRequest
    ) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper model."""
        try:
            # Transcribe
            options = {
                "language": None if request.language == "auto" else request.language,
                "temperature": settings.WHISPER_TEMPERATURE,
                "word_timestamps": True,
            }
            
            result = self.model.transcribe(audio_array, **options)
            
            # Extract word-level timestamps
            words_list = []
            if "segments" in result:
                for segment in result["segments"]:
                    if "words" in segment:
                        for word in segment["words"]:
                            words_list.append({
                                "word": word.get("word", ""),
                                "start": word.get("start", 0.0),
                                "end": word.get("end", 0.0),
                                "probability": word.get("probability", 0.0)
                            })
            
            # Calculate duration from audio length
            duration = len(audio_array) / settings.AUDIO_SAMPLE_RATE
            
            return {
                "text": result["text"].strip(),
                "confidence": 0.8,  # OpenAI Whisper doesn't provide confidence scores
                "language": result.get("language", "en"),
                "duration": duration,
                "words": words_list if words_list else None
            }
            
        except Exception as e:
            logger.error(f"OpenAI Whisper transcription failed: {e}")
            raise
    
    async def transcribe_streaming(self, audio_chunks: list) -> AsyncIterable[str]:
        """
        Transcribe audio in streaming mode.
        
        Args:
            audio_chunks: List of audio chunk bytes
            
        Yields:
            Partial transcription results
        """
        if not self.is_initialized:
            raise RuntimeError("Transcription service not initialized")
        
        accumulated_audio = b""
        
        for chunk in audio_chunks:
            accumulated_audio += chunk
            
            # Process when we have enough audio (e.g., 2 seconds)
            if len(accumulated_audio) > settings.AUDIO_SAMPLE_RATE * 2 * 2:  # 2 seconds of 16-bit audio
                try:
                    # Create temporary request
                    request = TranscriptionRequest(
                        audio_data=accumulated_audio,
                        language="auto"
                    )
                    
                    # Transcribe
                    result = await self.transcribe(request)
                    
                    if result.success and result.text.strip():
                        yield result.text
                    
                    # Reset accumulated audio (or keep overlap for context)
                    accumulated_audio = accumulated_audio[-settings.AUDIO_SAMPLE_RATE:]  # Keep 0.5 seconds overlap
                    
                except Exception as e:
                    logger.error(f"Streaming transcription error: {e}")
                    continue
    
    def is_ready(self) -> bool:
        """Check if the service is ready."""
        return self.is_initialized and (self.model is not None or self.fast_model is not None)
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up transcription service...")
        
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.fast_model is not None:
                del self.fast_model
                self.fast_model = None
            
            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_initialized = False
            logger.info("Transcription service cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during transcription service cleanup: {e}")
    
    async def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        # Common languages supported by Whisper
        return [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh",
            "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi", "cs",
            "sk", "hu", "ro", "bg", "hr", "sl", "et", "lv", "lt", "mt"
        ]
    
    async def detect_language(self, audio_data: bytes) -> str:
        """
        Detect the language of the audio.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Detected language code
        """
        try:
            audio_array = await self._preprocess_audio(audio_data)
            
            if self.model_type == "faster_whisper" and self.fast_model:
                _, info = self.fast_model.transcribe(
                    audio_array[:settings.AUDIO_SAMPLE_RATE * 30],  # Use first 30 seconds
                    language=None  # Auto-detect
                )
                return info.language if hasattr(info, 'language') else "en"
            else:
                result = self.model.transcribe(
                    audio_array[:settings.AUDIO_SAMPLE_RATE * 30],
                    language=None
                )
                return result.get("language", "en")
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en"  # Default to English 