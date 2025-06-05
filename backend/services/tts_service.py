"""
Text-to-Speech Service using XTTS-v2

Handles voice synthesis with support for:
- Voice cloning using XTTS-v2
- Multiple languages
- Real-time speech generation
- Custom voice training
- Voice quality optimization
"""

import asyncio
import logging
import time
import tempfile
import os
import io
from typing import Optional, Dict, Any, List, AsyncIterable
from pathlib import Path
import uuid

import torch
import torchaudio
import numpy as np
from TTS.api import TTS
import soundfile as sf
from pydub import AudioSegment

from ..models.schemas import TTSRequest, TTSResponse
from ..config import settings

logger = logging.getLogger(__name__)


class TTSService:
    """Service for converting text to speech using XTTS-v2."""
    
    def __init__(self):
        self.tts_model = None
        self.device = None
        self.is_initialized = False
        self.voice_samples = {}
        self.default_speaker = "default"
        self.supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr", 
            "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu"
        ]
        
    async def initialize(self):
        """Initialize the TTS model."""
        logger.info("Initializing TTS service...")
        
        try:
            # Determine device
            self.device = self._get_device()
            logger.info(f"Using device: {self.device}")
            
            # Initialize XTTS model
            await self._load_xtts_model()
            
            # Load voice samples if available
            await self._load_voice_samples()
            
            self.is_initialized = True
            logger.info("TTS service initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {e}")
            raise
    
    def _get_device(self) -> str:
        """Determine the best device to use."""
        if torch.cuda.is_available() and settings.USE_GPU:
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    async def _load_xtts_model(self):
        """Load XTTS-v2 model."""
        try:
            # Initialize TTS with XTTS-v2
            self.tts_model = TTS(
                model_name=settings.TTS_MODEL,
                progress_bar=False,
                gpu=self.device != "cpu"
            )
            
            if self.tts_model.is_multi_speaker:
                logger.info("XTTS-v2 model supports multi-speaker synthesis")
            
            if self.tts_model.is_multi_lingual:
                logger.info("XTTS-v2 model supports multi-lingual synthesis")
            
            logger.info("XTTS-v2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load XTTS-v2 model: {e}")
            
            # Fallback to a simpler TTS model
            try:
                logger.info("Attempting to load fallback TTS model...")
                self.tts_model = TTS(
                    model_name="tts_models/en/ljspeech/tacotron2-DDC",
                    progress_bar=False,
                    gpu=self.device != "cpu"
                )
                logger.info("Fallback TTS model loaded successfully")
                
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback TTS model: {fallback_error}")
                raise
    
    async def _load_voice_samples(self):
        """Load voice samples for cloning."""
        voice_samples_dir = settings.VOICE_SAMPLES_DIR
        
        if not voice_samples_dir.exists():
            logger.info("No voice samples directory found, using default voice")
            return
        
        # Look for audio files in voice samples directory
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        voice_files = []
        
        for ext in audio_extensions:
            voice_files.extend(voice_samples_dir.glob(f"*{ext}"))
        
        if voice_files:
            logger.info(f"Found {len(voice_files)} voice sample files")
            
            # Use the first voice sample as default speaker reference
            default_voice_path = str(voice_files[0])
            self.voice_samples[self.default_speaker] = default_voice_path
            
            logger.info(f"Default voice sample: {default_voice_path}")
        else:
            logger.info("No voice sample files found in voice samples directory")
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Convert text to speech.
        
        Args:
            request: TTS request with text and voice settings
            
        Returns:
            TTSResponse with synthesized audio
        """
        if not self.is_initialized:
            raise RuntimeError("TTS service not initialized")
        
        start_time = time.time()
        
        try:
            # Clean and prepare text
            text = self._clean_text(request.text)
            
            if not text.strip():
                raise ValueError("Empty text provided")
            
            # Generate speech
            audio_data = await self._generate_speech(request, text)
            
            # Save audio to temporary file for URL access
            audio_url = await self._save_audio_temp(audio_data, request.format)
            
            processing_time = time.time() - start_time
            
            # Calculate duration
            duration = len(audio_data) / request.sample_rate
            
            response = TTSResponse(
                audio_data=audio_data,
                audio_url=audio_url,
                duration=duration,
                format=request.format,
                sample_rate=request.sample_rate,
                processing_time=processing_time
            )
            
            logger.debug(f"TTS synthesis completed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            processing_time = time.time() - start_time
            
            return TTSResponse(
                duration=0.0,
                format=request.format,
                sample_rate=request.sample_rate,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for TTS."""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Handle common abbreviations and symbols
        replacements = {
            "&": " and ",
            "@": " at ",
            "#": " hash ",
            "%": " percent ",
            "$": " dollar ",
            "€": " euro ",
            "£": " pound ",
            "¥": " yen ",
            "+": " plus ",
            "=": " equals ",
            "<": " less than ",
            ">": " greater than ",
            "|": " or ",
            "\\": " backslash ",
            "/": " slash ",
            "~": " tilde ",
            "`": " backtick ",
            "^": " caret ",
            "*": " star ",
            "()": "",
            "[]": "",
            "{}": "",
        }
        
        for symbol, replacement in replacements.items():
            text = text.replace(symbol, replacement)
        
        # Ensure proper sentence endings
        if text and not text.endswith(('.', '!', '?', ':')):
            text += '.'
        
        return text
    
    async def _generate_speech(self, request: TTSRequest, text: str) -> bytes:
        """Generate speech audio from text."""
        try:
            # Check if we have voice cloning capability
            if (hasattr(self.tts_model, 'is_multi_speaker') and 
                self.tts_model.is_multi_speaker and 
                request.voice_id in self.voice_samples):
                
                # Use voice cloning
                return await self._generate_cloned_speech(request, text)
            else:
                # Use standard TTS
                return await self._generate_standard_speech(request, text)
                
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            raise
    
    async def _generate_cloned_speech(self, request: TTSRequest, text: str) -> bytes:
        """Generate speech with voice cloning."""
        try:
            speaker_wav = self.voice_samples.get(request.voice_id, self.voice_samples.get(self.default_speaker))
            
            if not speaker_wav:
                raise ValueError(f"No voice sample found for voice_id: {request.voice_id}")
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
                output_path = temp_output.name
            
            try:
                # Generate speech with voice cloning
                self.tts_model.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=request.language,
                    file_path=output_path,
                    speed=request.speed
                )
                
                # Read generated audio
                audio_data = await self._read_audio_file(output_path)
                
                # Apply pitch modification if needed
                if request.pitch != 1.0:
                    audio_data = self._adjust_pitch(audio_data, request.sample_rate, request.pitch)
                
                return audio_data
                
            finally:
                # Clean up temporary file
                Path(output_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            raise
    
    async def _generate_standard_speech(self, request: TTSRequest, text: str) -> bytes:
        """Generate speech without voice cloning."""
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
                output_path = temp_output.name
            
            try:
                # Check if model supports the requested language
                if hasattr(self.tts_model, 'is_multi_lingual') and self.tts_model.is_multi_lingual:
                    # Multi-lingual model
                    self.tts_model.tts_to_file(
                        text=text,
                        language=request.language,
                        file_path=output_path
                    )
                else:
                    # Single language model
                    self.tts_model.tts_to_file(
                        text=text,
                        file_path=output_path
                    )
                
                # Read generated audio
                audio_data = await self._read_audio_file(output_path)
                
                # Apply speed and pitch modifications
                if request.speed != 1.0 or request.pitch != 1.0:
                    audio_data = self._modify_audio(
                        audio_data, 
                        request.sample_rate, 
                        request.speed, 
                        request.pitch
                    )
                
                return audio_data
                
            finally:
                # Clean up temporary file
                Path(output_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Standard TTS failed: {e}")
            raise
    
    async def _read_audio_file(self, file_path: str) -> bytes:
        """Read audio file and return as bytes."""
        try:
            # Load audio with soundfile
            audio_array, sample_rate = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Normalize audio
            audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Convert to bytes
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio_array, sample_rate, format='WAV')
            
            return audio_bytes.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to read audio file: {e}")
            raise
    
    def _adjust_pitch(self, audio_data: bytes, sample_rate: int, pitch_factor: float) -> bytes:
        """Adjust pitch of audio data."""
        try:
            # Load audio from bytes
            audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
            
            # Adjust pitch by changing playback rate
            new_sample_rate = int(sample_rate * pitch_factor)
            pitched_audio = audio_segment._spawn(audio_segment.raw_data, overrides={
                "frame_rate": new_sample_rate
            }).set_frame_rate(sample_rate)
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            pitched_audio.export(output_buffer, format="wav")
            
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Pitch adjustment failed: {e}")
            return audio_data  # Return original audio if adjustment fails
    
    def _modify_audio(
        self, 
        audio_data: bytes, 
        sample_rate: int, 
        speed: float, 
        pitch: float
    ) -> bytes:
        """Apply speed and pitch modifications to audio."""
        try:
            # Load audio from bytes
            audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
            
            # Adjust speed
            if speed != 1.0:
                # Speed up or slow down
                audio_segment = audio_segment.speedup(playback_speed=speed)
            
            # Adjust pitch
            if pitch != 1.0:
                new_sample_rate = int(sample_rate * pitch)
                audio_segment = audio_segment._spawn(
                    audio_segment.raw_data, 
                    overrides={"frame_rate": new_sample_rate}
                ).set_frame_rate(sample_rate)
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            audio_segment.export(output_buffer, format="wav")
            
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Audio modification failed: {e}")
            return audio_data  # Return original audio if modification fails
    
    async def _save_audio_temp(self, audio_data: bytes, format_type) -> str:
        """Save audio data to temporary file and return URL."""
        try:
            # Create temporary file
            temp_dir = settings.TEMP_DIR
            temp_dir.mkdir(exist_ok=True)
            
            file_id = str(uuid.uuid4())
            file_extension = format_type.value if hasattr(format_type, 'value') else str(format_type)
            temp_file_path = temp_dir / f"tts_{file_id}.{file_extension}"
            
            # Write audio data
            with open(temp_file_path, 'wb') as f:
                f.write(audio_data)
            
            # Return URL path (relative to static serving)
            return f"/static/temp/tts_{file_id}.{file_extension}"
            
        except Exception as e:
            logger.error(f"Failed to save temporary audio file: {e}")
            return None
    
    async def train_voice(self, voice_samples: List[bytes], speaker_name: str) -> Dict[str, Any]:
        """
        Train a new voice model from samples.
        
        Args:
            voice_samples: List of audio sample bytes
            speaker_name: Name for the new voice
            
        Returns:
            Training result information
        """
        try:
            # Save voice samples to temporary files
            sample_paths = []
            
            for i, sample_data in enumerate(voice_samples):
                temp_path = settings.TEMP_DIR / f"voice_sample_{speaker_name}_{i}.wav"
                
                with open(temp_path, 'wb') as f:
                    f.write(sample_data)
                
                sample_paths.append(str(temp_path))
            
            # For XTTS-v2, we typically just need one good quality sample
            # Use the first sample as the reference
            if sample_paths:
                self.voice_samples[speaker_name] = sample_paths[0]
                
                logger.info(f"Voice '{speaker_name}' registered with sample: {sample_paths[0]}")
                
                return {
                    "voice_id": speaker_name,
                    "status": "ready",
                    "sample_path": sample_paths[0],
                    "message": "Voice registered successfully"
                }
            else:
                raise ValueError("No voice samples provided")
                
        except Exception as e:
            logger.error(f"Voice training failed: {e}")
            return {
                "voice_id": speaker_name,
                "status": "error",
                "error": str(e)
            }
    
    async def get_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices."""
        voices = []
        
        # Add default voices
        voices.append({
            "voice_id": "default",
            "name": "Default Voice",
            "language": "en",
            "type": "default"
        })
        
        # Add custom trained voices
        for voice_id, sample_path in self.voice_samples.items():
            if voice_id != "default":
                voices.append({
                    "voice_id": voice_id,
                    "name": voice_id.replace("_", " ").title(),
                    "language": "auto",
                    "type": "custom",
                    "sample_path": sample_path
                })
        
        return voices
    
    async def synthesize_streaming(self, text: str, voice_id: str = "default") -> AsyncIterable[bytes]:
        """
        Generate streaming audio synthesis.
        
        Args:
            text: Text to synthesize
            voice_id: Voice to use
            
        Yields:
            Audio chunks as bytes
        """
        try:
            # Split text into sentences for streaming
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                if sentence.strip():
                    request = TTSRequest(
                        text=sentence,
                        voice_id=voice_id,
                        language="en"
                    )
                    
                    result = await self.synthesize(request)
                    
                    if result.success and result.audio_data:
                        yield result.audio_data
                        
        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming synthesis."""
        import re
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Ensure sentences end with punctuation
        sentences = [s + '.' if not s.endswith(('.', '!', '?')) else s for s in sentences]
        
        return sentences
    
    def is_ready(self) -> bool:
        """Check if the service is ready."""
        return self.is_initialized and self.tts_model is not None
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up TTS service...")
        
        try:
            if self.tts_model is not None:
                del self.tts_model
                self.tts_model = None
            
            # Clear voice samples
            self.voice_samples.clear()
            
            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_initialized = False
            logger.info("TTS service cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during TTS service cleanup: {e}")
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.supported_languages.copy()
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current TTS model."""
        if not self.is_initialized:
            return {"error": "Service not initialized"}
        
        try:
            return {
                "initialized": self.is_initialized,
                "model_name": settings.TTS_MODEL,
                "device": self.device,
                "is_multi_speaker": getattr(self.tts_model, 'is_multi_speaker', False),
                "is_multi_lingual": getattr(self.tts_model, 'is_multi_lingual', False),
                "supported_languages": self.supported_languages,
                "available_voices": len(self.voice_samples),
                "voice_cloning_enabled": len(self.voice_samples) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)} 