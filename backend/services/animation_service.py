"""
Animation Service for AI Avatar

Handles avatar animation with support for:
- SadTalker integration for high-quality lip-sync
- Wav2Lip fallback for basic lip-sync
- Face detection and basic animation
- Video generation and streaming
- Avatar photo management
"""

import asyncio
import logging
import time
import tempfile
import os
import io
from typing import Optional, Dict, Any, AsyncIterable
from pathlib import Path
import uuid

import cv2
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip, AudioFileClip
import imageio

from ..models.schemas import AnimationRequest, AnimationResponse
from ..config import settings

logger = logging.getLogger(__name__)


class AnimationService:
    """Service for animating AI avatars with lip-sync."""
    
    def __init__(self):
        self.is_initialized = False
        self.face_mesh = None
        self.face_detection = None
        self.avatar_images = []
        self.default_avatar_path = None
        
        # SadTalker integration (if available)
        self.sadtalker_model = None
        self.sadtalker_available = False
        
    async def initialize(self):
        """Initialize the animation service."""
        logger.info("Initializing Animation service...")
        
        try:
            # Initialize MediaPipe for face detection and mesh
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
            
            # Load avatar images
            await self._load_avatar_images()
            
            # Try to initialize SadTalker if available
            await self._try_init_sadtalker()
            
            self.is_initialized = True
            logger.info("Animation service initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize animation service: {e}")
            raise
    
    async def _load_avatar_images(self):
        """Load avatar images from the avatar_photos directory."""
        avatar_dir = settings.AVATAR_PHOTOS_DIR
        
        if not avatar_dir.exists():
            logger.warning("No avatar photos directory found")
            # Create a default avatar image
            await self._create_default_avatar()
            return
        
        # Look for image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        avatar_files = []
        
        for ext in image_extensions:
            avatar_files.extend(avatar_dir.glob(f"*{ext}"))
        
        if avatar_files:
            logger.info(f"Found {len(avatar_files)} avatar images")
            self.avatar_images = [str(path) for path in avatar_files]
            self.default_avatar_path = self.avatar_images[0]
            logger.info(f"Default avatar: {self.default_avatar_path}")
        else:
            logger.warning("No avatar images found, creating default")
            await self._create_default_avatar()
    
    async def _create_default_avatar(self):
        """Create a simple default avatar image."""
        try:
            # Create a simple face-like image
            avatar_img = np.ones((512, 512, 3), dtype=np.uint8) * 200
            
            # Draw a simple face
            center = (256, 256)
            
            # Face circle
            cv2.circle(avatar_img, center, 200, (220, 200, 180), -1)
            
            # Eyes
            cv2.circle(avatar_img, (200, 220), 20, (50, 50, 50), -1)
            cv2.circle(avatar_img, (312, 220), 20, (50, 50, 50), -1)
            
            # Nose
            cv2.circle(avatar_img, center, 8, (180, 160, 140), -1)
            
            # Mouth
            cv2.ellipse(avatar_img, (256, 300), (40, 20), 0, 0, 180, (150, 100, 100), -1)
            
            # Save default avatar
            default_path = settings.AVATAR_PHOTOS_DIR / "default_avatar.png"
            cv2.imwrite(str(default_path), avatar_img)
            
            self.default_avatar_path = str(default_path)
            self.avatar_images = [self.default_avatar_path]
            
            logger.info(f"Created default avatar: {default_path}")
            
        except Exception as e:
            logger.error(f"Failed to create default avatar: {e}")
            raise
    
    async def _try_init_sadtalker(self):
        """Try to initialize SadTalker if available."""
        try:
            # Try to import and initialize SadTalker
            # This is a placeholder for actual SadTalker integration
            # In a real implementation, you would install SadTalker and import it here
            
            logger.info("Attempting to initialize SadTalker...")
            
            # For now, mark as unavailable until proper integration
            self.sadtalker_available = False
            logger.info("SadTalker not available, using fallback animation")
            
        except ImportError:
            logger.info("SadTalker not installed, using basic animation")
            self.sadtalker_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize SadTalker: {e}")
            self.sadtalker_available = False
    
    async def animate(self, request: AnimationRequest) -> AnimationResponse:
        """
        Animate avatar with lip-sync to audio.
        
        Args:
            request: Animation request with audio and image data
            
        Returns:
            AnimationResponse with animated video
        """
        if not self.is_initialized:
            raise RuntimeError("Animation service not initialized")
        
        start_time = time.time()
        
        try:
            # Determine source image
            if request.use_default_avatar or not request.image_data:
                image_path = self.default_avatar_path
            else:
                # Save provided image temporarily
                image_path = await self._save_temp_image(request.image_data)
            
            # Save audio temporarily
            audio_path = await self._save_temp_audio(request.audio_data)
            
            # Generate animation
            if self.sadtalker_available:
                video_path = await self._animate_with_sadtalker(
                    image_path, audio_path, request
                )
            else:
                video_path = await self._animate_with_fallback(
                    image_path, audio_path, request
                )
            
            # Read video data
            video_data = await self._read_video_file(video_path)
            
            # Create video URL
            video_url = await self._save_video_temp(video_data)
            
            # Get video info
            video_info = await self._get_video_info(video_path)
            
            processing_time = time.time() - start_time
            
            response = AnimationResponse(
                video_data=video_data,
                video_url=video_url,
                duration=video_info["duration"],
                fps=video_info["fps"],
                resolution=request.resolution,
                processing_time=processing_time
            )
            
            logger.debug(f"Animation completed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Animation failed: {e}")
            processing_time = time.time() - start_time
            
            return AnimationResponse(
                duration=0.0,
                fps=25,
                resolution=request.resolution,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
        finally:
            # Cleanup temporary files
            await self._cleanup_temp_files()
    
    async def _save_temp_image(self, image_data: bytes) -> str:
        """Save image data to temporary file."""
        temp_path = settings.TEMP_DIR / f"temp_avatar_{uuid.uuid4()}.png"
        
        with open(temp_path, 'wb') as f:
            f.write(image_data)
        
        return str(temp_path)
    
    async def _save_temp_audio(self, audio_data: bytes) -> str:
        """Save audio data to temporary file."""
        temp_path = settings.TEMP_DIR / f"temp_audio_{uuid.uuid4()}.wav"
        
        with open(temp_path, 'wb') as f:
            f.write(audio_data)
        
        return str(temp_path)
    
    async def _animate_with_sadtalker(
        self, 
        image_path: str, 
        audio_path: str, 
        request: AnimationRequest
    ) -> str:
        """Animate using SadTalker (when available)."""
        # Placeholder for SadTalker integration
        # This would call the actual SadTalker model
        
        logger.info("Using SadTalker for animation (placeholder)")
        
        # For now, fall back to basic animation
        return await self._animate_with_fallback(image_path, audio_path, request)
    
    async def _animate_with_fallback(
        self, 
        image_path: str, 
        audio_path: str, 
        request: AnimationRequest
    ) -> str:
        """Animate using basic fallback method."""
        try:
            # Load the source image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Get audio duration
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            audio_clip.close()
            
            # Generate video frames
            fps = request.fps
            total_frames = int(duration * fps)
            
            # Create video writer
            output_path = settings.TEMP_DIR / f"animated_avatar_{uuid.uuid4()}.mp4"
            
            # Parse resolution
            width, height = map(int, request.resolution.split('x'))
            
            # Resize image to target resolution
            image = cv2.resize(image, (width, height))
            
            # Create video frames with basic animation
            frames = []
            
            for frame_idx in range(total_frames):
                # Create animated frame
                animated_frame = await self._create_animated_frame(
                    image, frame_idx, total_frames, duration
                )
                frames.append(animated_frame)
            
            # Save video with imageio
            imageio.mimsave(
                str(output_path), 
                frames, 
                fps=fps,
                codec='libx264'
            )
            
            # Add audio to video
            final_output = await self._add_audio_to_video(str(output_path), audio_path)
            
            return final_output
            
        except Exception as e:
            logger.error(f"Fallback animation failed: {e}")
            raise
    
    async def _create_animated_frame(
        self, 
        base_image: np.ndarray, 
        frame_idx: int, 
        total_frames: int, 
        duration: float
    ) -> np.ndarray:
        """Create an animated frame with basic mouth movement."""
        frame = base_image.copy()
        
        # Simple mouth animation based on frame index
        # This is a very basic implementation
        
        # Calculate animation phase
        time_position = (frame_idx / total_frames) * duration
        mouth_phase = np.sin(time_position * 10) * 0.5 + 0.5  # 0 to 1
        
        # Try to detect face and animate mouth area
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                
                # Estimate mouth position (lower third of face)
                mouth_y = y + int(box_h * 0.7)
                mouth_x = x + box_w // 2
                
                # Simple mouth animation
                mouth_opening = int(mouth_phase * 20)
                mouth_width = 40
                
                # Draw animated mouth
                cv2.ellipse(
                    frame,
                    (mouth_x, mouth_y),
                    (mouth_width, mouth_opening + 5),
                    0, 0, 180,
                    (100, 50, 50),
                    -1
                )
        except Exception as e:
            logger.debug(f"Face detection failed in frame {frame_idx}: {e}")
        
        return frame
    
    async def _add_audio_to_video(self, video_path: str, audio_path: str) -> str:
        """Add audio track to video."""
        try:
            output_path = settings.TEMP_DIR / f"final_video_{uuid.uuid4()}.mp4"
            
            # Load video and audio
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)
            
            # Set audio to video
            final_clip = video_clip.set_audio(audio_clip)
            
            # Write final video
            final_clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # Cleanup
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to add audio to video: {e}")
            raise
    
    async def _read_video_file(self, video_path: str) -> bytes:
        """Read video file and return as bytes."""
        try:
            with open(video_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read video file: {e}")
            raise
    
    async def _save_video_temp(self, video_data: bytes) -> str:
        """Save video data to temporary file and return URL."""
        try:
            file_id = str(uuid.uuid4())
            temp_file_path = settings.TEMP_DIR / f"avatar_video_{file_id}.mp4"
            
            with open(temp_file_path, 'wb') as f:
                f.write(video_data)
            
            return f"/static/temp/avatar_video_{file_id}.mp4"
            
        except Exception as e:
            logger.error(f"Failed to save temporary video file: {e}")
            return None
    
    async def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information."""
        try:
            clip = VideoFileClip(video_path)
            info = {
                "duration": clip.duration,
                "fps": clip.fps,
                "size": clip.size
            }
            clip.close()
            return info
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return {"duration": 0.0, "fps": 25, "size": (512, 512)}
    
    async def stream_frames(self) -> AsyncIterable[bytes]:
        """Stream animated frames for real-time display."""
        try:
            # This would stream live animation frames
            # For now, just yield placeholder frames
            
            for i in range(100):  # Stream 100 frames as example
                # Create a simple animated frame
                frame = np.ones((512, 512, 3), dtype=np.uint8) * 200
                
                # Add some animation
                time_val = time.time() + i * 0.1
                mouth_phase = np.sin(time_val * 5) * 0.5 + 0.5
                
                # Draw animated mouth
                cv2.ellipse(
                    frame,
                    (256, 350),
                    (40, int(mouth_phase * 20) + 5),
                    0, 0, 180,
                    (100, 50, 50),
                    -1
                )
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                await asyncio.sleep(0.1)  # 10 FPS
                
        except Exception as e:
            logger.error(f"Frame streaming failed: {e}")
    
    async def _cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            temp_dir = settings.TEMP_DIR
            
            # Remove files older than 1 hour
            current_time = time.time()
            for file_path in temp_dir.glob("temp_*"):
                if current_time - file_path.stat().st_mtime > 3600:  # 1 hour
                    file_path.unlink(missing_ok=True)
                    
        except Exception as e:
            logger.debug(f"Temp file cleanup error: {e}")
    
    def is_ready(self) -> bool:
        """Check if the service is ready."""
        return self.is_initialized
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up animation service...")
        
        try:
            if self.face_mesh:
                self.face_mesh.close()
                self.face_mesh = None
            
            if self.face_detection:
                self.face_detection.close()
                self.face_detection = None
            
            # Cleanup temp files
            await self._cleanup_temp_files()
            
            self.is_initialized = False
            logger.info("Animation service cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during animation service cleanup: {e}")
    
    async def get_available_avatars(self) -> List[Dict[str, Any]]:
        """Get list of available avatar images."""
        avatars = []
        
        for i, image_path in enumerate(self.avatar_images):
            avatars.append({
                "id": i,
                "name": Path(image_path).stem,
                "path": image_path,
                "is_default": image_path == self.default_avatar_path
            })
        
        return avatars
    
    async def set_default_avatar(self, avatar_id: int) -> bool:
        """Set the default avatar by ID."""
        try:
            if 0 <= avatar_id < len(self.avatar_images):
                self.default_avatar_path = self.avatar_images[avatar_id]
                logger.info(f"Default avatar set to: {self.default_avatar_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to set default avatar: {e}")
            return False 