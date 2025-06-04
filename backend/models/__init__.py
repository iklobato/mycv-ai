"""
Models package for AI Avatar Application
"""

from .schemas import (
    # Request schemas
    TranscriptionRequest,
    LLMRequest,
    TTSRequest,
    AnimationRequest,
    VoiceTrainingRequest,
    AvatarTrainingRequest,
    
    # Response schemas
    TranscriptionResponse,
    LLMResponse,
    TTSResponse,
    AnimationResponse,
    VoiceTrainingResponse,
    AvatarTrainingResponse,
    
    # WebSocket message schemas
    WebSocketMessage,
    AudioChunkMessage,
    TranscriptionMessage,
    LLMResponseMessage,
    ErrorMessage,
    StatusMessage,
    
    # System schemas
    ModelInfo,
    SystemStatus,
    AudioConfig,
    VideoConfig,
    ModelConfig,
    
    # Conversation schemas
    ConversationTurn,
    Conversation,
    
    # Enums
    ModelStatus,
    AudioFormat,
    VideoFormat,
    
    # Validation functions
    validate_audio_data,
    validate_image_data,
)

__all__ = [
    "TranscriptionRequest",
    "LLMRequest",
    "TTSRequest",
    "AnimationRequest",
    "VoiceTrainingRequest",
    "AvatarTrainingRequest",
    "TranscriptionResponse",
    "LLMResponse",
    "TTSResponse",
    "AnimationResponse",
    "VoiceTrainingResponse",
    "AvatarTrainingResponse",
    "WebSocketMessage",
    "AudioChunkMessage",
    "TranscriptionMessage",
    "LLMResponseMessage",
    "ErrorMessage",
    "StatusMessage",
    "ModelInfo",
    "SystemStatus",
    "AudioConfig",
    "VideoConfig",
    "ModelConfig",
    "ConversationTurn",
    "Conversation",
    "ModelStatus",
    "AudioFormat",
    "VideoFormat",
    "validate_audio_data",
    "validate_image_data",
] 