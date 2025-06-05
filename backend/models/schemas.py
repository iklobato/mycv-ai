"""
Pydantic schemas for AI Avatar Application API
"""

from typing import Optional, List, Dict, Any, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class ModelStatus(str, Enum):
    """Model loading status."""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    M4A = "m4a"
    FLAC = "flac"


class VideoFormat(str, Enum):
    """Supported video formats."""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"


# Base schemas
class BaseRequest(BaseModel):
    """Base request schema."""
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


class BaseResponse(BaseModel):
    """Base response schema."""
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: Optional[float] = None
    error_message: Optional[str] = None


# Transcription schemas
class TranscriptionRequest(BaseModel):
    """Request schema for audio transcription."""
    audio_data: bytes = Field(..., description="Base64 encoded audio data")
    language: str = Field("auto", description="Language for transcription")
    response_format: str = Field("json", description="Response format")
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Temperature for transcription")


class TranscriptionWord(BaseModel):
    """Schema for a transcription word."""
    word: str = Field(..., description="The word")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class TranscriptionResponse(BaseModel):
    """Response schema for audio transcription."""
    text: str = Field(..., description="Transcribed text")
    language: str = Field(..., description="Detected language")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    word_count: int = Field(..., description="Number of words transcribed")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")
    words: Optional[List[TranscriptionWord]] = Field(None, description="Word-level timestamps")
    success: bool = Field(True, description="Whether the request was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")


# LLM schemas
class LLMRequest(BaseModel):
    """Request schema for LLM generation."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Response creativity")
    max_tokens: int = Field(512, ge=50, le=2048, description="Maximum tokens in response")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or whitespace only')
        return v.strip()


class LLMResponse(BaseModel):
    """Response schema for LLM generation."""
    response: str = Field(..., description="Generated response")
    model: str = Field(..., description="Model used for generation")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    processing_time: float = Field(..., description="Processing time in seconds")
    success: bool = Field(True, description="Whether the request was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    token_count: Optional[int] = Field(None, description="Number of tokens generated")


# TTS schemas
class TTSRequest(BaseModel):
    """Request schema for text-to-speech."""
    text: str = Field(min_length=1, max_length=5000)
    voice_id: str = "default"
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: float = Field(default=1.0, ge=0.5, le=2.0)
    format: AudioFormat = AudioFormat.WAV
    sample_rate: int = Field(default=22050, ge=8000, le=48000)


class TTSResponse(BaseResponse):
    """Response schema for text-to-speech."""
    audio_data: Optional[bytes] = None
    audio_url: Optional[str] = None
    duration: float
    format: AudioFormat
    sample_rate: int
    
    class Config:
        arbitrary_types_allowed = True


# Animation schemas
class AnimationRequest(BaseRequest):
    """Request schema for avatar animation."""
    audio_data: bytes
    image_data: Optional[bytes] = None
    use_default_avatar: bool = True
    audio_filename: Optional[str] = None
    image_filename: Optional[str] = None
    animation_style: str = "default"
    quality: str = "medium"  # low, medium, high
    fps: int = Field(default=25, ge=15, le=60)
    resolution: str = "512x512"
    
    class Config:
        arbitrary_types_allowed = True


class AnimationResponse(BaseResponse):
    """Response schema for avatar animation."""
    video_data: Optional[bytes] = None
    video_url: Optional[str] = None
    duration: float
    fps: int
    resolution: str
    format: VideoFormat = VideoFormat.MP4
    
    class Config:
        arbitrary_types_allowed = True


# WebSocket message schemas
class WebSocketMessage(BaseModel):
    """Base WebSocket message schema."""
    type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any]


class AudioChunkMessage(WebSocketMessage):
    """WebSocket message for audio chunks."""
    type: str = "audio_chunk"
    audio_data: bytes
    chunk_id: int
    is_final: bool = False
    
    class Config:
        arbitrary_types_allowed = True


class TranscriptionMessage(WebSocketMessage):
    """WebSocket message for transcription results."""
    type: str = "transcription"
    text: str
    confidence: float
    is_partial: bool = False


class LLMResponseMessage(WebSocketMessage):
    """WebSocket message for LLM responses."""
    type: str = "llm_response"
    text: str
    is_streaming: bool = False


class ErrorMessage(WebSocketMessage):
    """WebSocket message for errors."""
    type: str = "error"
    error_code: str
    error_message: str


class StatusMessage(WebSocketMessage):
    """WebSocket message for status updates."""
    type: str = "status"
    status: str
    message: str


# Model status schemas
class ModelInfo(BaseModel):
    """Information about a loaded model."""
    name: str
    status: ModelStatus
    version: Optional[str] = None
    device: Optional[str] = None
    memory_usage: Optional[float] = None
    load_time: Optional[float] = None
    error_message: Optional[str] = None


class SystemStatus(BaseModel):
    """Overall system status."""
    status: str
    uptime: float
    models: Dict[str, ModelInfo]
    gpu_available: bool
    gpu_memory_total: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None


# Configuration schemas
class AudioConfig(BaseModel):
    """Audio configuration settings."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 2.0
    format: AudioFormat = AudioFormat.WAV
    quality: str = "medium"


class VideoConfig(BaseModel):
    """Video configuration settings."""
    fps: int = 25
    resolution: str = "512x512"
    codec: str = "libx264"
    bitrate: str = "1M"
    format: VideoFormat = VideoFormat.MP4


class ModelConfig(BaseModel):
    """Model configuration settings."""
    whisper_model: str = "base"
    llm_model: str = "llama3"
    tts_model: str = "xtts_v2"
    animation_model: str = "sadtalker"
    use_gpu: bool = True
    batch_size: int = 1


# Training schemas (for voice cloning)
class VoiceTrainingRequest(BaseModel):
    """Request schema for voice training."""
    voice_samples: List[bytes]
    speaker_name: str
    language: str = "en"
    training_steps: int = Field(default=1000, ge=100, le=10000)
    
    class Config:
        arbitrary_types_allowed = True


class VoiceTrainingResponse(BaseResponse):
    """Response schema for voice training."""
    voice_id: str
    training_progress: float = Field(ge=0.0, le=1.0)
    is_complete: bool = False
    model_path: Optional[str] = None


# Avatar training schemas
class AvatarTrainingRequest(BaseModel):
    """Request schema for avatar training."""
    photos: List[bytes]
    avatar_name: str
    training_steps: int = Field(default=500, ge=50, le=5000)
    
    class Config:
        arbitrary_types_allowed = True


class AvatarTrainingResponse(BaseResponse):
    """Response schema for avatar training."""
    avatar_id: str
    training_progress: float = Field(ge=0.0, le=1.0)
    is_complete: bool = False
    model_path: Optional[str] = None


# Conversation schemas
class ConversationTurn(BaseModel):
    """Single turn in a conversation."""
    user_text: str
    ai_response: str
    timestamp: datetime = Field(default_factory=datetime.now)
    audio_url: Optional[str] = None
    video_url: Optional[str] = None


class Conversation(BaseModel):
    """Complete conversation history."""
    conversation_id: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    turns: List[ConversationTurn] = []
    metadata: Dict[str, Any] = {}


# Streaming schemas
class StreamingChunk(BaseModel):
    """Streaming response chunk."""
    chunk_id: int
    data: Union[str, bytes]
    is_final: bool = False
    content_type: str
    
    class Config:
        arbitrary_types_allowed = True


# Analytics schemas
class UsageStats(BaseModel):
    """Usage statistics."""
    total_requests: int = 0
    total_audio_processed: float = 0.0  # in seconds
    total_text_generated: int = 0  # in characters
    average_response_time: float = 0.0
    error_rate: float = 0.0
    uptime: float = 0.0


class PerformanceMetrics(BaseModel):
    """Performance metrics."""
    transcription_latency: float = 0.0
    llm_latency: float = 0.0
    tts_latency: float = 0.0
    animation_latency: float = 0.0
    end_to_end_latency: float = 0.0
    throughput: float = 0.0  # requests per second


# Validation functions
def validate_audio_data(audio_data: bytes) -> bytes:
    """Validate audio data format and size."""
    if len(audio_data) == 0:
        raise ValueError("Audio data cannot be empty")
    if len(audio_data) > 50 * 1024 * 1024:  # 50MB limit
        raise ValueError("Audio file too large")
    return audio_data


def validate_image_data(image_data: bytes) -> bytes:
    """Validate image data format and size."""
    if len(image_data) == 0:
        raise ValueError("Image data cannot be empty")
    if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
        raise ValueError("Image file too large")
    return image_data


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, bool] = Field(..., description="Individual service status")
    cv_loaded: bool = Field(..., description="Whether CV is loaded")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")


class CVInfoResponse(BaseModel):
    """Response model for CV information."""
    status: str = Field(..., description="CV service status")
    cv_info: Dict[str, Any] = Field(..., description="CV information")
    system_prompt_preview: Optional[str] = Field(None, description="Preview of system prompt")


class ModelStatusResponse(BaseModel):
    """Response model for model status."""
    cv: Dict[str, Any] = Field(..., description="CV service status")
    whisper: Dict[str, Any] = Field(..., description="Whisper model status")
    ollama: Dict[str, Any] = Field(..., description="Ollama service status")
    xtts: Dict[str, Any] = Field(..., description="XTTS model status")
    sadtalker: Dict[str, Any] = Field(..., description="SadTalker model status")


class WebSocketMessage(BaseModel):
    """Base model for WebSocket messages."""
    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.now)


class PingMessage(WebSocketMessage):
    """Ping message."""
    type: Literal["ping"] = "ping"


class PongMessage(WebSocketMessage):
    """Pong response message."""
    type: Literal["pong"] = "pong"


class TextInputMessage(WebSocketMessage):
    """Text input message."""
    type: Literal["text_input"] = "text_input"
    text: str = Field(..., description="Input text")


class AudioDataMessage(WebSocketMessage):
    """Audio data message."""
    type: Literal["audio_data"] = "audio_data"
    audio_data: str = Field(..., description="Base64 encoded audio data")
    format: str = Field("wav", description="Audio format")


class TranscriptionMessage(WebSocketMessage):
    """Transcription result message."""
    type: Literal["transcription"] = "transcription"
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., description="Confidence score")


class LLMResponseMessage(WebSocketMessage):
    """LLM response message."""
    type: Literal["llm_response"] = "llm_response"
    response: str = Field(..., description="Generated response")
    conversation_id: str = Field(..., description="Conversation ID")


class CompleteResponseMessage(WebSocketMessage):
    """Complete response message with all pipeline outputs."""
    type: Literal["complete_response"] = "complete_response"
    transcription: str = Field(..., description="Original transcription")
    llm_response: str = Field(..., description="LLM generated response")
    audio_url: str = Field(..., description="TTS audio URL")
    video_url: str = Field(..., description="Avatar animation URL")
    processing_time: float = Field(..., description="Total processing time")


class ErrorMessage(WebSocketMessage):
    """Error message."""
    type: Literal["error"] = "error"
    message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")


class ContextMessage(WebSocketMessage):
    """Conversation context message."""
    type: Literal["conversation_context"] = "conversation_context"
    context: List[Dict[str, str]] = Field(..., description="Conversation history")


class SettingsMessage(WebSocketMessage):
    """Settings update message."""
    type: Literal["settings"] = "settings"
    settings: Dict[str, Any] = Field(..., description="User settings")


class UserSettings(BaseModel):
    """User settings model."""
    voice_speed: float = Field(1.0, ge=0.5, le=2.0)
    voice_pitch: float = Field(1.0, ge=0.5, le=2.0)
    model: Optional[str] = Field(None)
    animation_style: str = Field("realistic")
    auto_animate: bool = Field(True)
    save_history: bool = Field(True) 