"""
AI Avatar FastAPI Application

Main application file that handles real-time AI avatar interactions including:
- Speech transcription via Whisper
- LLM responses via Ollama with CV-based personality
- Voice cloning via XTTS-v2
- Avatar animation via SadTalker
"""

import os
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Temporary stub services to avoid dependency issues
class TranscriptionService:
    async def initialize(self): pass
    async def cleanup(self): pass
    def is_ready(self): return False

class LLMService:
    async def initialize(self): pass
    async def cleanup(self): pass
    def is_ready(self): return False

class TTSService:
    async def initialize(self): pass
    async def cleanup(self): pass
    def is_ready(self): return False

class AnimationService:
    async def initialize(self): pass
    async def cleanup(self): pass
    def is_ready(self): return False

class WebSocketManager:
    def __init__(self): pass
    async def connect(self, websocket, connection_id): pass
    async def disconnect(self, connection_id): pass
    async def get_stats(self): return {}

from backend.services.cv_service import cv_service
from backend.models.schemas import (
    TranscriptionRequest,
    TranscriptionResponse,
    LLMRequest,
    LLMResponse,
    TTSRequest,
    TTSResponse,
    AnimationRequest,
    AnimationResponse,
)
# from backend.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Avatar Application",
    description="Real-time AI avatar interaction system with CV-based personality",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
transcription_service = TranscriptionService()
llm_service = LLMService()
tts_service = TTSService()
animation_service = AnimationService()
websocket_manager = WebSocketManager()

# Mount static files
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Initializing AI Avatar Application...")
    
    try:
        # Initialize CV service first (other services may depend on it)
        await cv_service.initialize()
        
        # Initialize all other services
        await transcription_service.initialize()
        await llm_service.initialize()
        await tts_service.initialize()
        await animation_service.initialize()
        
        logger.info("All services initialized successfully!")
        
        # Log CV status
        if cv_service.has_cv():
            logger.info(f"✅ CV loaded for {cv_service.default_name}")
        else:
            logger.warning("⚠️  No CV loaded - AI will use fallback personality")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down AI Avatar Application...")
    
    # Cleanup services
    await cv_service.cleanup()
    await transcription_service.cleanup()
    await llm_service.cleanup()
    await tts_service.cleanup()
    await animation_service.cleanup()


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main application page."""
    html_file = frontend_path / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    else:
        return HTMLResponse(
            content="<h1>AI Avatar App</h1><p>Frontend not found. Please build the frontend.</p>",
            status_code=404
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "cv": cv_service.is_ready(),
            "transcription": transcription_service.is_ready(),
            "llm": llm_service.is_ready(),
            "tts": tts_service.is_ready(),
            "animation": animation_service.is_ready(),
        },
        "cv_loaded": cv_service.has_cv() if cv_service.is_ready() else False
    }


@app.get("/cv")
async def get_cv_info():
    """
    Get CV information and status (for debugging).
    
    Returns:
        CV status, content preview, and configuration
    """
    try:
        if not cv_service.is_ready():
            raise HTTPException(status_code=503, detail="CV service not ready")
        
        cv_info = cv_service.get_cv_info()
        
        return {
            "status": "ready",
            "cv_info": cv_info,
            "system_prompt_preview": cv_service.get_system_prompt()[:300] + "..." if len(cv_service.get_system_prompt()) > 300 else cv_service.get_system_prompt()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CV info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cv/reload")
async def reload_cv():
    """
    Reload CV content from file.
    
    Returns:
        Success status and updated CV info
    """
    try:
        if not cv_service.is_ready():
            raise HTTPException(status_code=503, detail="CV service not ready")
        
        success = await cv_service.reload_cv()
        
        if success:
            cv_info = cv_service.get_cv_info()
            return {
                "status": "success",
                "message": "CV reloaded successfully",
                "cv_info": cv_info
            }
        else:
            return {
                "status": "error",
                "message": "Failed to reload CV"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CV reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language: str = Form("auto")
):
    """
    Transcribe audio to text using Whisper.
    
    Args:
        audio_file: Audio file to transcribe
        language: Target language (auto-detect if 'auto')
    
    Returns:
        TranscriptionResponse with transcribed text
    """
    try:
        # Read audio data
        audio_data = await audio_file.read()
        
        # Create request
        request = TranscriptionRequest(
            audio_data=audio_data,
            language=language,
            filename=audio_file.filename
        )
        
        # Transcribe
        result = await transcription_service.transcribe(request)
        
        logger.info(f"Transcribed: {result.text[:100]}...")
        return result
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/respond", response_model=LLMResponse)
async def generate_response(request: LLMRequest):
    """
    Generate AI response using local LLM with CV-based personality.
    
    Args:
        request: LLM request with user text and context
    
    Returns:
        LLMResponse with generated text (as Henrique Lobato)
    """
    try:
        # Generate response (CV context automatically injected by LLM service)
        result = await llm_service.generate_response(request)
        
        logger.info(f"Generated response: {result.response[:100]}...")
        return result
        
    except Exception as e:
        logger.error(f"LLM response error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speak", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """
    Convert text to speech using voice cloning.
    
    Args:
        request: TTS request with text and voice settings
    
    Returns:
        TTSResponse with audio data
    """
    try:
        # Synthesize speech
        result = await tts_service.synthesize(request)
        
        logger.info(f"Synthesized speech for: {request.text[:50]}...")
        return result
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/animate", response_model=AnimationResponse)
async def animate_avatar(
    audio_file: UploadFile = File(...),
    image_file: UploadFile = File(None),
    use_default_avatar: bool = Form(True)
):
    """
    Animate avatar with lip-sync to audio.
    
    Args:
        audio_file: Audio file for lip-sync
        image_file: Optional custom avatar image
        use_default_avatar: Whether to use the default trained avatar
    
    Returns:
        AnimationResponse with animated video
    """
    try:
        # Read audio data
        audio_data = await audio_file.read()
        
        # Read image data if provided
        image_data = None
        if image_file and not use_default_avatar:
            image_data = await image_file.read()
        
        # Create request
        request = AnimationRequest(
            audio_data=audio_data,
            image_data=image_data,
            use_default_avatar=use_default_avatar,
            audio_filename=audio_file.filename,
            image_filename=image_file.filename if image_file else None
        )
        
        # Animate
        result = await animation_service.animate(request)
        
        logger.info("Avatar animation completed")
        return result
        
    except Exception as e:
        logger.error(f"Animation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/avatar/stream")
async def stream_avatar():
    """Stream live avatar animation."""
    async def generate_frames():
        try:
            async for frame in animation_service.stream_frames():
                yield frame
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            return
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time communication.
    
    Handles real-time audio streaming, transcription, LLM responses,
    TTS, and avatar animation in a continuous pipeline. All LLM responses
    will use CV-based personality automatically.
    """
    connection_id = await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Wait for message
            message = await websocket.receive()
            
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    # Handle audio data
                    audio_data = message["bytes"]
                    await websocket_manager.stream_audio(connection_id, audio_data)
                    await process_audio_pipeline(websocket, connection_id, audio_data)
                    
                elif "text" in message:
                    # Handle text messages (commands, settings, etc.)
                    try:
                        data = json.loads(message["text"])
                        await handle_text_message(websocket, connection_id, data)
                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid JSON format"
                        })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass  # Connection might be closed
        websocket_manager.disconnect(websocket)


async def handle_text_message(websocket: WebSocket, connection_id: str, data: dict):
    """
    Handle text messages from WebSocket clients.
    
    Args:
        websocket: WebSocket connection
        connection_id: Connection identifier
        data: Parsed JSON data
    """
    message_type = data.get("type")
    
    if message_type == "ping":
        # Handle ping/pong for keepalive
        await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
        
    elif message_type == "clear_context":
        # Clear conversation context
        websocket_manager.clear_conversation_context(connection_id)
        await websocket.send_json({"type": "context_cleared"})
        
    elif message_type == "get_context":
        # Send conversation context
        context = websocket_manager.get_conversation_context(connection_id)
        await websocket.send_json({
            "type": "conversation_context",
            "context": context
        })
        
    elif message_type == "text_input":
        # Handle direct text input (without audio)
        text = data.get("text", "").strip()
        if text:
            await process_text_pipeline(websocket, connection_id, text)
            
    elif message_type == "settings":
        # Handle settings updates
        settings_data = data.get("settings", {})
        await update_user_settings(connection_id, settings_data)
        await websocket.send_json({"type": "settings_updated"})
        
    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        })


async def process_audio_pipeline(websocket: WebSocket, connection_id: str, audio_data: bytes):
    """
    Process the complete audio pipeline:
    Audio -> Transcription -> LLM (with CV context) -> TTS -> Animation
    
    Args:
        websocket: WebSocket connection
        connection_id: Connection identifier
        audio_data: Raw audio bytes
    """
    try:
        # Step 1: Transcribe audio
        transcription_request = TranscriptionRequest(
            audio_data=audio_data,
            language="auto"
        )
        transcription_result = await transcription_service.transcribe(transcription_request)
        
        # Send transcription to client
        await websocket.send_json({
            "type": "transcription",
            "text": transcription_result.text,
            "confidence": transcription_result.confidence,
            "connection_id": connection_id
        })
        
        # Skip if no speech detected
        if not transcription_result.text.strip():
            return
            
        # Add user message to conversation context
        websocket_manager.add_conversation_message(
            connection_id, "user", transcription_result.text
        )
        
        # Process the transcribed text
        await process_text_pipeline(websocket, connection_id, transcription_result.text)
        
    except Exception as e:
        logger.error(f"Audio pipeline error for {connection_id}: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Audio processing failed: {str(e)}"
        })


async def process_text_pipeline(websocket: WebSocket, connection_id: str, user_text: str):
    """
    Process text through LLM (with CV context) -> TTS -> Animation pipeline.
    
    Args:
        websocket: WebSocket connection
        connection_id: Connection identifier
        user_text: User's text input
    """
    try:
        # Get conversation context
        conversation_context = websocket_manager.get_conversation_context(connection_id)
        
        # Step 2: Generate LLM response (CV context automatically injected)
        llm_request = LLMRequest(
            message=user_text,
            conversation_id=connection_id
        )
        llm_result = await llm_service.generate_response(llm_request)
        
        # Add assistant response to conversation context
        websocket_manager.add_conversation_message(
            connection_id, "assistant", llm_result.response
        )
        
        # Send LLM response to client
        await websocket.send_json({
            "type": "llm_response",
            "text": llm_result.response,
            "connection_id": connection_id
        })
        
        # Step 3: Synthesize speech
        tts_request = TTSRequest(
            text=llm_result.response,
            voice_id="default",
            speed=1.0,
            pitch=1.0
        )
        tts_result = await tts_service.synthesize(tts_request)
        
        # Step 4: Animate avatar
        animation_request = AnimationRequest(
            audio_data=tts_result.audio_data,
            use_default_avatar=True
        )
        animation_result = await animation_service.animate(animation_request)
        
        # Send complete response to client
        await websocket.send_json({
            "type": "complete_response",
            "transcription": user_text,
            "llm_response": llm_result.response,
            "audio_url": tts_result.audio_url,
            "video_url": animation_result.video_url,
            "duration": animation_result.duration,
            "connection_id": connection_id
        })
        
    except Exception as e:
        logger.error(f"Text pipeline error for {connection_id}: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Text processing failed: {str(e)}"
        })


async def update_user_settings(connection_id: str, settings_data: dict):
    """
    Update user settings for a connection.
    
    Args:
        connection_id: Connection identifier
        settings_data: Settings to update
    """
    # This could store settings in the WebSocket manager's session_data
    # For now, just log the settings
    logger.info(f"Updated settings for {connection_id}: {settings_data}")


# Add new endpoint for WebSocket connection statistics
@app.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    return websocket_manager.get_stats()


@app.get("/models/status")
async def get_models_status():
    """Get status of all AI models including CV integration."""
    return {
        "cv": {
            "loaded": cv_service.is_ready(),
            "has_content": cv_service.has_cv() if cv_service.is_ready() else False,
            "file_path": str(cv_service.cv_file_path) if cv_service.is_ready() else None,
        },
        "whisper": {
            "loaded": transcription_service.is_ready(),
            "model": "whisper-base",
        },
        "ollama": {
            "loaded": llm_service.is_ready(),
            "model": "llama3.2:latest",
            "base_url": "http://localhost:11434",
            "cv_integration": True,
        },
        "xtts": {
            "loaded": tts_service.is_ready(),
            "voice_model": "xtts-v2",
        },
        "sadtalker": {
            "loaded": animation_service.is_ready(),
            "model_path": "sadtalker",
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    ) 