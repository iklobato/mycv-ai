"""
Unit tests for the AnimationService.

Tests cover:
- Service initialization and configuration
- Avatar animation and lip-sync functionality
- Video generation and processing
- Face detection and animation
- Error handling and edge cases
- Avatar management and streaming
"""

import pytest
import tempfile
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, mock_open
from pathlib import Path
import numpy as np

# Import the service (mocks are set up in conftest.py)
from backend.services.animation_service import AnimationService
from backend.models.schemas import AnimationRequest, AnimationResponse


class TestAnimationService:
    """Test suite for AnimationService."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.service = AnimationService()
    
    def teardown_method(self):
        """Clean up after each test."""
        pass
    
    def test_animation_service_initialization(self):
        """Test AnimationService initialization."""
        service = AnimationService()
        
        assert service.is_initialized is False
        assert service.face_mesh is None
        assert service.face_detection is None
        assert service.avatar_images == []
        assert service.default_avatar_path is None
        assert service.sadtalker_model is None
        assert service.sadtalker_available is False
    
    async def test_initialize_success(self):
        """Test successful service initialization."""
        with patch('backend.services.animation_service.settings') as mock_settings:
            mock_settings.AVATAR_PHOTOS_DIR = Path("/tmp/avatars")
            
            with patch('backend.services.animation_service.mp') as mock_mp, \
                 patch.object(self.service, '_load_avatar_images') as mock_load_avatars, \
                 patch.object(self.service, '_try_init_sadtalker') as mock_init_sadtalker:
                
                # Mock MediaPipe components
                mock_mp.solutions.face_mesh.FaceMesh.return_value = Mock()
                mock_mp.solutions.face_detection.FaceDetection.return_value = Mock()
                
                await self.service.initialize()
                
                assert self.service.is_initialized is True
                assert self.service.face_mesh is not None
                assert self.service.face_detection is not None
                mock_load_avatars.assert_called_once()
                mock_init_sadtalker.assert_called_once()
    
    async def test_initialize_failure(self):
        """Test service initialization failure."""
        with patch.object(self.service, '_load_avatar_images', side_effect=Exception("Avatar loading failed")):
            with pytest.raises(Exception, match="Avatar loading failed"):
                await self.service.initialize()
    
    async def test_load_avatar_images_with_files(self):
        """Test loading avatar images when files exist."""
        with patch('backend.services.animation_service.settings') as mock_settings:
            avatar_dir = Path("/tmp/avatars")
            mock_settings.AVATAR_PHOTOS_DIR = avatar_dir
            
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.glob') as mock_glob:
                
                # Mock Path.glob to return different files for different patterns
                def glob_side_effect(pattern):
                    if pattern == "*.jpg":
                        return [Path("/tmp/avatars/avatar1.jpg")]
                    elif pattern == "*.png":
                        return [Path("/tmp/avatars/avatar2.png")]
                    else:
                        return []
                
                mock_glob.side_effect = glob_side_effect
                
                await self.service._load_avatar_images()
                
                assert len(self.service.avatar_images) == 2
                assert self.service.default_avatar_path == str(Path("/tmp/avatars/avatar1.jpg"))
    
    async def test_load_avatar_images_no_directory(self):
        """Test loading avatar images when directory doesn't exist."""
        with patch('backend.services.animation_service.settings') as mock_settings:
            avatar_dir = Path("/tmp/nonexistent")
            mock_settings.AVATAR_PHOTOS_DIR = avatar_dir
            
            with patch('pathlib.Path.exists', return_value=False), \
                 patch.object(self.service, '_create_default_avatar') as mock_create:
                
                await self.service._load_avatar_images()
                
                mock_create.assert_called_once()
    
    async def test_create_default_avatar_success(self):
        """Test creating default avatar."""
        with patch('backend.services.animation_service.settings') as mock_settings:
            mock_settings.AVATAR_PHOTOS_DIR = Path("/tmp/avatars")
            
            with patch('backend.services.animation_service.np.ones') as mock_ones, \
                 patch('backend.services.animation_service.cv2.imwrite') as mock_imwrite:
                
                mock_ones.return_value = [[255, 255, 255]]
                mock_imwrite.return_value = True
                
                await self.service._create_default_avatar()
                
                assert self.service.default_avatar_path is not None
                assert len(self.service.avatar_images) == 1
                mock_imwrite.assert_called_once()
    
    async def test_try_init_sadtalker_not_available(self):
        """Test SadTalker initialization when not available."""
        await self.service._try_init_sadtalker()
        
        assert self.service.sadtalker_available is False
    
    async def test_animate_success_with_default_avatar(self):
        """Test successful animation with default avatar."""
        self.service.is_initialized = True
        self.service.default_avatar_path = "/tmp/default_avatar.png"
        
        with patch.object(self.service, '_save_temp_audio', return_value="/tmp/audio.wav") as mock_save_audio, \
             patch.object(self.service, '_animate_with_fallback', return_value="/tmp/video.mp4") as mock_animate, \
             patch.object(self.service, '_read_video_file', return_value=b"video_data") as mock_read_video, \
             patch.object(self.service, '_save_video_temp', return_value="http://test.com/video.mp4") as mock_save_video, \
             patch.object(self.service, '_get_video_info') as mock_video_info:
            
            mock_video_info.return_value = {
                "duration": 3.0,
                "fps": 25,
                "resolution": "512x512"
            }
            
            request = AnimationRequest(
                audio_data=b"fake_audio_data",
                use_default_avatar=True
            )
            
            response = await self.service.animate(request)
            
            assert response.success is True
            assert response.video_data == b"video_data"
            assert response.video_url == "http://test.com/video.mp4"
            assert response.duration == 3.0
            assert response.processing_time > 0
            
            mock_save_audio.assert_called_once()
            mock_animate.assert_called_once()
    
    async def test_animate_success_with_custom_image(self):
        """Test successful animation with custom image."""
        self.service.is_initialized = True
        
        with patch.object(self.service, '_save_temp_image', return_value="/tmp/image.jpg") as mock_save_image, \
             patch.object(self.service, '_save_temp_audio', return_value="/tmp/audio.wav") as mock_save_audio, \
             patch.object(self.service, '_animate_with_fallback', return_value="/tmp/video.mp4") as mock_animate, \
             patch.object(self.service, '_read_video_file', return_value=b"video_data") as mock_read_video, \
             patch.object(self.service, '_save_video_temp', return_value="http://test.com/video.mp4") as mock_save_video, \
             patch.object(self.service, '_get_video_info') as mock_video_info:
            
            mock_video_info.return_value = {
                "duration": 3.0,
                "fps": 25,
                "resolution": "512x512"
            }
            
            request = AnimationRequest(
                audio_data=b"fake_audio_data",
                image_data=b"fake_image_data",
                use_default_avatar=False
            )
            
            response = await self.service.animate(request)
            
            assert response.success is True
            mock_save_image.assert_called_once()
    
    async def test_animate_not_initialized(self):
        """Test animation when service not initialized."""
        request = AnimationRequest(
            audio_data=b"fake_audio_data",
            use_default_avatar=True
        )
        
        with pytest.raises(RuntimeError, match="Animation service not initialized"):
            await self.service.animate(request)
    
    async def test_animate_exception_handling(self):
        """Test animation exception handling."""
        self.service.is_initialized = True
        self.service.default_avatar_path = "/tmp/default_avatar.png"
        
        with patch.object(self.service, '_save_temp_audio', side_effect=Exception("Audio save failed")):
            request = AnimationRequest(
                audio_data=b"fake_audio_data",
                use_default_avatar=True
            )
            
            response = await self.service.animate(request)
            
            assert response.success is False
            assert "Audio save failed" in response.error_message
    
    async def test_save_temp_image_success(self):
        """Test saving temporary image."""
        image_data = b"fake_image_data"
        
        with patch('backend.services.animation_service.settings') as mock_settings:
            mock_settings.TEMP_DIR = Path("/tmp")
            
            result = await self.service._save_temp_image(image_data)
            
            # Check that it returns a valid path
            assert result.startswith("/tmp/temp_avatar_")
            assert result.endswith(".png")
    
    async def test_save_temp_audio_success(self):
        """Test saving temporary audio."""
        audio_data = b"fake_audio_data"
        
        with patch('backend.services.animation_service.settings') as mock_settings:
            mock_settings.TEMP_DIR = Path("/tmp")
            
            result = await self.service._save_temp_audio(audio_data)
            
            # Check that it returns a valid path
            assert result.startswith("/tmp/temp_audio_")
            assert result.endswith(".wav")
    
    async def test_animate_with_sadtalker_not_available(self):
        """Test animation with SadTalker when not available."""
        self.service.sadtalker_available = False
        
        with patch.object(self.service, '_animate_with_fallback', return_value="/tmp/video.mp4") as mock_fallback:
            request = AnimationRequest(
                audio_data=b"fake_audio_data",
                use_default_avatar=True
            )
            
            result = await self.service._animate_with_sadtalker("/tmp/image.jpg", "/tmp/audio.wav", request)
            
            # Should call fallback method when SadTalker is not available
            assert result == "/tmp/video.mp4"
            mock_fallback.assert_called_once()
    
    async def test_animate_with_fallback_success(self):
        """Test fallback animation method."""
        self.service.face_mesh = Mock()
        
        with patch('backend.services.animation_service.cv2.imread') as mock_imread, \
             patch('backend.services.animation_service.AudioFileClip') as mock_audio_clip, \
             patch('backend.services.animation_service.imageio.mimsave') as mock_mimsave, \
             patch.object(self.service, '_create_animated_frame') as mock_create_frame, \
             patch.object(self.service, '_add_audio_to_video', return_value="/tmp/final_video.mp4") as mock_add_audio:
            
            # Mock image and audio loading
            mock_imread.return_value = np.ones((512, 512, 3), dtype=np.uint8) * 200
            
            # Mock AudioFileClip
            mock_audio = Mock()
            mock_audio.duration = 3.0
            mock_audio_clip.return_value = mock_audio
            
            # Mock frame creation
            mock_create_frame.return_value = np.ones((512, 512, 3), dtype=np.uint8) * 200
            
            request = AnimationRequest(
                audio_data=b"fake_audio_data",
                use_default_avatar=True
            )
            
            result = await self.service._animate_with_fallback("/tmp/image.jpg", "/tmp/audio.wav", request)
            
            assert result == "/tmp/final_video.mp4"
            mock_add_audio.assert_called_once()
    
    async def test_create_animated_frame_success(self):
        """Test creating animated frame."""
        base_image = [[255, 255, 255]]
        
        with patch('backend.services.animation_service.np.sin') as mock_sin:
            mock_sin.return_value = 0.5
            
            result = await self.service._create_animated_frame(base_image, 10, 100, 2.0)
            
            assert result is not None
    
    async def test_add_audio_to_video_success(self):
        """Test adding audio to video."""
        with patch('backend.services.animation_service.VideoFileClip') as mock_video_clip, \
             patch('backend.services.animation_service.AudioFileClip') as mock_audio_clip:
            
            # Mock video and audio clips
            mock_video = Mock()
            mock_audio = Mock()
            mock_final = Mock()
            
            mock_video_clip.return_value = mock_video
            mock_audio_clip.return_value = mock_audio
            mock_video.set_audio.return_value = mock_final
            mock_final.write_videofile.return_value = None
            
            result = await self.service._add_audio_to_video("/tmp/video.mp4", "/tmp/audio.wav")
            
            assert result.endswith(".mp4")
            mock_final.write_videofile.assert_called_once()
    
    async def test_read_video_file_success(self):
        """Test reading video file."""
        with patch('builtins.open', mock_open(read_data=b"video_content")) as mock_file:
            result = await self.service._read_video_file("/tmp/video.mp4")
            
            assert result == b"video_content"
            mock_file.assert_called_once_with("/tmp/video.mp4", 'rb')
    
    async def test_save_video_temp_success(self):
        """Test saving video to temporary file."""
        video_data = b"fake_video_data"
        
        with patch('backend.services.animation_service.settings') as mock_settings:
            mock_settings.TEMP_DIR = Path("/tmp")
            
            result = await self.service._save_video_temp(video_data)
            
            # Check that it returns a valid URL
            assert result.startswith("/static/temp/avatar_video_")
            assert result.endswith(".mp4")
    
    async def test_get_video_info_success(self):
        """Test getting video information."""
        with patch('backend.services.animation_service.VideoFileClip') as mock_video_clip:
            mock_clip = Mock()
            mock_clip.duration = 3.0
            mock_clip.fps = 25
            mock_clip.size = (512, 512)
            mock_video_clip.return_value = mock_clip
            
            result = await self.service._get_video_info("/tmp/video.mp4")
            
            assert result["duration"] == 3.0
            assert result["fps"] == 25
            assert result["size"] == (512, 512)
            mock_clip.close.assert_called_once()
    
    async def test_stream_frames_success(self):
        """Test streaming video frames."""
        self.service.is_initialized = True
        self.service.default_avatar_path = "/tmp/default_avatar.png"
        
        with patch('backend.services.animation_service.cv2.imencode') as mock_encode, \
             patch('backend.services.animation_service.time.time', return_value=0), \
             patch('backend.services.animation_service.asyncio.sleep', new_callable=AsyncMock):
            
            mock_encode.return_value = (True, np.array([1, 2, 3, 4, 5], dtype=np.uint8))
            
            frames = []
            # Limit the stream to avoid infinite loop in test
            count = 0
            async for frame in self.service.stream_frames():
                frames.append(frame)
                count += 1
                if count >= 2:  # Just test first 2 frames
                    break
            
            assert len(frames) == 2
            # Check that frames have the HTTP streaming format
            for frame in frames:
                assert b'--frame' in frame
                assert b'Content-Type: image/jpeg' in frame
    
    async def test_cleanup_temp_files_success(self):
        """Test cleaning up temporary files."""
        with patch('backend.services.animation_service.os.remove') as mock_remove, \
             patch('backend.services.animation_service.os.path.exists', return_value=True):
            
            await self.service._cleanup_temp_files()
            
            # Should complete without error (temp files list is empty by default)
    
    def test_is_ready_true(self):
        """Test is_ready when service is initialized."""
        self.service.is_initialized = True
        assert self.service.is_ready() is True
    
    def test_is_ready_false(self):
        """Test is_ready when service is not initialized."""
        self.service.is_initialized = False
        assert self.service.is_ready() is False
    
    async def test_cleanup_success(self):
        """Test successful cleanup."""
        self.service.face_mesh = Mock()
        self.service.face_detection = Mock()
        self.service.is_initialized = True
        
        with patch.object(self.service, '_cleanup_temp_files') as mock_cleanup_files:
            await self.service.cleanup()
            
            assert self.service.face_mesh is None
            assert self.service.face_detection is None
            assert self.service.is_initialized is False
            mock_cleanup_files.assert_called_once()
    
    async def test_cleanup_with_exception(self):
        """Test cleanup with exception handling."""
        self.service.face_mesh = Mock()
        self.service.face_mesh.close.side_effect = Exception("Cleanup failed")
        self.service.is_initialized = True
        
        # Should not raise exception due to error handling
        await self.service.cleanup()
        
        # Should still set is_initialized to False despite exception (this is the correct behavior)
        assert self.service.is_initialized is False
    
    async def test_get_available_avatars_success(self):
        """Test getting available avatars."""
        self.service.avatar_images = ["/tmp/avatar1.jpg", "/tmp/avatar2.png"]
        
        avatars = await self.service.get_available_avatars()
        
        assert len(avatars) == 2
        assert avatars[0]["id"] == 0
        assert avatars[1]["id"] == 1
        assert "path" in avatars[0]
        assert "name" in avatars[0]
    
    async def test_set_default_avatar_success(self):
        """Test setting default avatar."""
        self.service.avatar_images = ["/tmp/avatar1.jpg", "/tmp/avatar2.png"]
        
        result = await self.service.set_default_avatar(1)
        
        assert result is True
        assert self.service.default_avatar_path == "/tmp/avatar2.png"
    
    async def test_set_default_avatar_invalid_id(self):
        """Test setting default avatar with invalid ID."""
        self.service.avatar_images = ["/tmp/avatar1.jpg"]
        
        result = await self.service.set_default_avatar(5)
        
        assert result is False 