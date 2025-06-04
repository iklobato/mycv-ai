"""
Configuration settings for AI Avatar Application
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    APP_NAME: str = "AI Avatar Application"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    AVATAR_PHOTOS_DIR: Path = BASE_DIR / "avatar_photos"
    VOICE_SAMPLES_DIR: Path = BASE_DIR / "voice_samples"
    TEMP_DIR: Path = BASE_DIR / "temp"
    STATIC_DIR: Path = BASE_DIR / "static"
    
    # Ollama LLM settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 512
    LLM_TIMEOUT: int = 30
    
    # Whisper settings
    WHISPER_MODEL: str = "base"  # tiny, base, small, medium, large
    WHISPER_DEVICE: str = "auto"  # auto, cpu, cuda
    WHISPER_LANGUAGE: str = "auto"
    WHISPER_TEMPERATURE: float = 0.0
    
    # TTS settings
    TTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    VOICE_MODEL_PATH: Optional[str] = None
    TTS_LANGUAGE: str = "en"
    TTS_SPEAKER: str = "default"
    TTS_SPEED: float = 1.0
    TTS_PITCH: float = 1.0
    
    # Avatar animation settings
    AVATAR_MODEL_PATH: Optional[str] = None
    ANIMATION_MODEL: str = "sadtalker"  # sadtalker, wav2lip
    ANIMATION_QUALITY: str = "medium"  # low, medium, high
    ANIMATION_FPS: int = 25
    ANIMATION_RESOLUTION: str = "512x512"
    
    # Audio settings
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    AUDIO_CHUNK_DURATION: float = 2.0  # seconds
    AUDIO_FORMAT: str = "wav"
    AUDIO_BITRATE: int = 128000
    
    # Video settings
    VIDEO_CODEC: str = "libx264"
    VIDEO_BITRATE: str = "1M"
    VIDEO_FPS: int = 25
    VIDEO_RESOLUTION: str = "512x512"
    
    # WebSocket settings
    WS_MAX_CONNECTIONS: int = 100
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MESSAGE_QUEUE_SIZE: int = 1000
    
    # Performance settings
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 60
    BATCH_SIZE: int = 1
    USE_GPU: bool = True
    GPU_MEMORY_FRACTION: float = 0.8
    
    # Security settings
    ALLOWED_HOSTS: list = ["*"]
    CORS_ORIGINS: list = ["*"]
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_AUDIO_FORMATS: list = ["wav", "mp3", "ogg", "m4a", "flac"]
    ALLOWED_IMAGE_FORMATS: list = ["jpg", "jpeg", "png", "bmp"]
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    LOG_ROTATION: str = "1 day"
    LOG_RETENTION: str = "30 days"
    
    # Cache settings
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # seconds
    CACHE_MAX_SIZE: int = 1000
    
    # Model-specific settings
    MODELS_CONFIG: dict = {
        "whisper": {
            "model_size": "base",
            "compute_type": "float16",
            "device": "auto",
            "num_workers": 1,
        },
        "xtts": {
            "temperature": 0.75,
            "length_penalty": 1.0,
            "repetition_penalty": 5.0,
            "top_k": 50,
            "top_p": 0.85,
        },
        "sadtalker": {
            "pose_style": 0,
            "exp_scale": 1.23,
            "use_ref_video": False,
            "ref_video": None,
            "ref_info": None,
            "use_idle_mode": False,
            "length_of_audio": 0,
            "use_blink": True,
        },
        "ollama": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_ctx": 2048,
            "num_predict": 512,
        }
    }
    
    @validator("BASE_DIR", "MODELS_DIR", "AVATAR_PHOTOS_DIR", "VOICE_SAMPLES_DIR", "TEMP_DIR", "STATIC_DIR")
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("WHISPER_DEVICE")
    def validate_device(cls, v):
        if v == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return v
    
    @validator("USE_GPU")
    def validate_gpu_usage(cls, v):
        if v:
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class DevelopmentSettings(Settings):
    """Development-specific settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    

class ProductionSettings(Settings):
    """Production-specific settings."""
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    ALLOWED_HOSTS: list = ["localhost", "127.0.0.1"]
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8000"]


def get_settings() -> Settings:
    """Get settings based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    else:
        return DevelopmentSettings()


# Global settings instance
settings = get_settings()

# Create necessary directories
for dir_path in [
    settings.MODELS_DIR,
    settings.AVATAR_PHOTOS_DIR,
    settings.VOICE_SAMPLES_DIR,
    settings.TEMP_DIR,
    settings.STATIC_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True) 