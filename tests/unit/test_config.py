"""
Unit tests for configuration module.

Tests cover:
- Settings validation and creation
- Environment-specific configurations
- Import error handling
- Path validation
- Device detection
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from backend.config import (
    Settings, DevelopmentSettings, ProductionSettings, 
    get_settings
)


class TestSettings:
    """Test base Settings class."""
    
    def test_settings_creation(self):
        """Test basic settings creation."""
        settings = Settings()
        assert settings.APP_NAME == "AI Avatar Application"
        assert settings.VERSION == "1.0.0"
        assert settings.DEBUG is False
        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 8000
    
    def test_settings_with_env_vars(self):
        """Test settings with environment variables."""
        with patch.dict(os.environ, {
            'DEBUG': 'True',
            'HOST': '127.0.0.1',
            'PORT': '9000'
        }):
            settings = Settings()
            assert settings.DEBUG is True
            assert settings.HOST == "127.0.0.1"
            assert settings.PORT == 9000


class TestSettingsValidators:
    """Test settings validators."""
    
    def test_create_directories_validator(self):
        """Test directory creation validator."""
        settings = Settings()
        # Directories should be created as Path objects
        assert isinstance(settings.BASE_DIR, Path)
        assert isinstance(settings.MODELS_DIR, Path)
        assert isinstance(settings.TEMP_DIR, Path)
    
    def test_create_directories_validator_with_string(self):
        """Test directory creation validator with string input."""
        # Test that string paths are converted to Path objects
        string_path = "/tmp/test_path"
        settings = Settings(TEMP_DIR=string_path)
        assert isinstance(settings.TEMP_DIR, Path)
        assert str(settings.TEMP_DIR) == string_path
    
    def test_validate_device_auto_with_torch_available(self):
        """Test device validation when torch is available and CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            settings = Settings(WHISPER_DEVICE="auto")
            assert settings.WHISPER_DEVICE == "cuda"
    
    def test_validate_device_auto_with_torch_no_cuda(self):
        """Test device validation when torch is available but CUDA is not."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            settings = Settings(WHISPER_DEVICE="auto")
            assert settings.WHISPER_DEVICE == "cpu"
    
    def test_validate_device_auto_import_error(self):
        """Test device validation when torch import fails."""
        # Remove torch from sys.modules if present and test ImportError path
        original_torch = sys.modules.pop('torch', None)
        try:
            settings = Settings(WHISPER_DEVICE="auto")
            assert settings.WHISPER_DEVICE == "cpu"
        finally:
            # Restore original torch if it existed
            if original_torch is not None:
                sys.modules['torch'] = original_torch
    
    def test_validate_device_manual(self):
        """Test device validation with manual setting."""
        settings = Settings(WHISPER_DEVICE="cpu")
        assert settings.WHISPER_DEVICE == "cpu"
    
    def test_validate_gpu_usage_true_with_torch(self):
        """Test GPU validation when USE_GPU is True and torch is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            settings = Settings(USE_GPU=True)
            assert settings.USE_GPU is True
    
    def test_validate_gpu_usage_true_no_cuda(self):
        """Test GPU validation when USE_GPU is True but CUDA is not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            settings = Settings(USE_GPU=True)
            assert settings.USE_GPU is False
    
    def test_validate_gpu_usage_import_error(self):
        """Test GPU validation when torch import fails."""
        # Remove torch from sys.modules if present and test ImportError path
        original_torch = sys.modules.pop('torch', None)
        try:
            settings = Settings(USE_GPU=True)
            assert settings.USE_GPU is False
        finally:
            # Restore original torch if it existed
            if original_torch is not None:
                sys.modules['torch'] = original_torch
    
    def test_validate_gpu_usage_false(self):
        """Test GPU validation when USE_GPU is False."""
        settings = Settings(USE_GPU=False)
        assert settings.USE_GPU is False

    def test_directory_path_string_conversion(self):
        """Test directory path validator with string path conversion."""
        from backend.config import Settings
        
        # Test with string path that triggers the Path conversion
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create settings with string path
            settings_data = {
                "BASE_DIR": temp_dir,  # This should trigger the string to Path conversion
                "MODELS_DIR": f"{temp_dir}/models",
                "AVATAR_PHOTOS_DIR": f"{temp_dir}/photos",
                "VOICE_SAMPLES_DIR": f"{temp_dir}/voice",
                "TEMP_DIR": f"{temp_dir}/temp",
                "STATIC_DIR": f"{temp_dir}/static"
            }
            
            # This should trigger the create_directories validator and line 140
            settings = Settings(**settings_data)
            
            # Verify the paths were converted and directories created
            from pathlib import Path
            assert isinstance(settings.BASE_DIR, Path)
            assert settings.BASE_DIR.exists()
    
    def test_string_to_path_conversion_direct(self):
        """Test direct string to Path conversion in create_directories validator."""
        from backend.config import Settings
        
        # Use a temporary directory as a string to trigger the conversion
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Test with just BASE_DIR as string - this should hit line 140
            settings = Settings(BASE_DIR=temp_dir)  # String input
            
            from pathlib import Path
            assert isinstance(settings.BASE_DIR, Path)
            assert str(settings.BASE_DIR) == temp_dir
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_create_directories_validator_coverage(self):
        """Test to ensure line 140 coverage for string to Path conversion."""
        from backend.config import Settings
        from pathlib import Path
        import tempfile
        
        # Create a temporary directory as a string
        temp_dir = tempfile.mkdtemp()
        try:
            # Test that triggers the string-to-Path conversion on line 140
            # Use environment variable to force string input
            import os
            with patch.dict(os.environ, {'BASE_DIR': temp_dir}):  # String input via env var
                settings = Settings()
                
                # Verify it was converted to Path
                assert isinstance(settings.BASE_DIR, Path)
                assert settings.BASE_DIR.exists()
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_create_directories_with_string_path_direct(self):
        """Test create_directories validator with explicit string path to hit line 140."""
        from backend.config import Settings
        from pathlib import Path
        import tempfile
        
        # Create settings with explicit string path for BASE_DIR
        temp_dir = tempfile.mkdtemp()
        try:
            # Force Pydantic to process a string input by creating Settings with string
            # This should go through the validator and hit line 140: v = Path(v)
            settings = Settings(BASE_DIR=temp_dir)  # String input
            
            assert isinstance(settings.BASE_DIR, Path)
            assert str(settings.BASE_DIR) == temp_dir
            assert settings.BASE_DIR.exists()
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_create_directories_validator_string_conversion(self):
        """Test that the validator actually converts strings to Paths on line 140."""
        from backend.config import Settings
        from pathlib import Path
        import tempfile
        
        # Test using a model rebuild approach that forces validation
        temp_dir = tempfile.mkdtemp()
        try:
            # Create model data with string paths to force validator execution
            model_data = {
                'BASE_DIR': temp_dir,  # String that should trigger Path(v) on line 140
                'MODELS_DIR': f"{temp_dir}/models",
                'TEMP_DIR': f"{temp_dir}/temp",
                'STATIC_DIR': f"{temp_dir}/static", 
                'AVATAR_PHOTOS_DIR': f"{temp_dir}/photos",
                'VOICE_SAMPLES_DIR': f"{temp_dir}/voice"
            }
            
            # Create Settings instance with string paths
            settings = Settings(**model_data)
            
            # Verify all paths were converted from strings to Path objects
            assert isinstance(settings.BASE_DIR, Path)
            assert isinstance(settings.MODELS_DIR, Path)
            assert isinstance(settings.TEMP_DIR, Path)
            
            # Verify directories were created
            assert settings.BASE_DIR.exists()
            assert settings.MODELS_DIR.exists()
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestEnvironmentSettings:
    """Test environment-specific settings."""
    
    def test_development_settings(self):
        """Test development settings."""
        settings = DevelopmentSettings()
        assert settings.DEBUG is True
        assert settings.LOG_LEVEL == "DEBUG"
    
    def test_production_settings(self):
        """Test production settings."""
        settings = ProductionSettings()
        assert settings.DEBUG is False
        assert settings.LOG_LEVEL == "WARNING"
        assert "localhost" in settings.ALLOWED_HOSTS
        assert "127.0.0.1" in settings.ALLOWED_HOSTS
    
    def test_get_settings_development(self):
        """Test get_settings for development environment."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            settings = get_settings()
            assert isinstance(settings, DevelopmentSettings)
            assert settings.DEBUG is True
    
    def test_get_settings_production(self):
        """Test get_settings for production environment."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            settings = get_settings()
            assert isinstance(settings, ProductionSettings)
            assert settings.DEBUG is False
    
    def test_get_settings_default(self):
        """Test get_settings with no environment set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ENVIRONMENT if it exists
            if 'ENVIRONMENT' in os.environ:
                del os.environ['ENVIRONMENT']
            settings = get_settings()
            assert isinstance(settings, DevelopmentSettings)
            assert settings.DEBUG is True
    
    def test_get_settings_unknown_environment(self):
        """Test get_settings with unknown environment."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'unknown'}):
            settings = get_settings()
            assert isinstance(settings, DevelopmentSettings)
            assert settings.DEBUG is True


class TestSettingsConfiguration:
    """Test specific settings configurations."""
    
    def test_models_config(self):
        """Test models configuration."""
        settings = Settings()
        assert "whisper" in settings.MODELS_CONFIG
        assert "xtts" in settings.MODELS_CONFIG
        assert "sadtalker" in settings.MODELS_CONFIG
        assert "ollama" in settings.MODELS_CONFIG
        
        # Test whisper config
        whisper_config = settings.MODELS_CONFIG["whisper"]
        assert whisper_config["model_size"] == "base"
        assert whisper_config["device"] == "auto"
    
    def test_path_settings(self):
        """Test path settings."""
        settings = Settings()
        assert settings.BASE_DIR.is_absolute()
        assert str(settings.MODELS_DIR).endswith("models")
        assert str(settings.TEMP_DIR).endswith("temp")
    
    def test_audio_video_settings(self):
        """Test audio and video settings."""
        settings = Settings()
        assert settings.AUDIO_SAMPLE_RATE == 16000
        assert settings.AUDIO_CHANNELS == 1
        assert settings.VIDEO_FPS == 25
        assert settings.VIDEO_CODEC == "libx264"
    
    def test_security_settings(self):
        """Test security settings."""
        settings = Settings()
        assert settings.MAX_FILE_SIZE == 50 * 1024 * 1024  # 50MB
        assert "wav" in settings.ALLOWED_AUDIO_FORMATS
        assert "mp3" in settings.ALLOWED_AUDIO_FORMATS
        assert "jpg" in settings.ALLOWED_IMAGE_FORMATS
        assert "png" in settings.ALLOWED_IMAGE_FORMATS 