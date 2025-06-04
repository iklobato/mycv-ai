import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from backend.models.schemas import LLMRequest, TranscriptionRequest, TTSRequest


class TestMainApp:
    
    @pytest.mark.unit
    def test_get_index_success(self, fastapi_test_client):
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value="<html><body>AI Avatar App</body></html>"):
            
            response = fastapi_test_client.get("/")
            
            assert response.status_code == 200
            assert "AI Avatar App" in response.text
    
    @pytest.mark.unit
    def test_get_index_not_found(self, fastapi_test_client):
        with patch('pathlib.Path.exists', return_value=False):
            
            response = fastapi_test_client.get("/")
            
            assert response.status_code == 404
            assert "Frontend not found" in response.text
    
    @pytest.mark.unit
    def test_health_check(self, fastapi_test_client):
        with patch('backend.main.cv_service') as mock_cv, \
             patch('backend.main.transcription_service') as mock_trans, \
             patch('backend.main.llm_service') as mock_llm, \
             patch('backend.main.tts_service') as mock_tts, \
             patch('backend.main.animation_service') as mock_anim:
            
            mock_cv.is_ready.return_value = True
            mock_cv.has_cv.return_value = True
            mock_trans.is_ready.return_value = True
            mock_llm.is_ready.return_value = True
            mock_tts.is_ready.return_value = True
            mock_anim.is_ready.return_value = True
            
            response = fastapi_test_client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["services"]["cv"] is True
            assert data["services"]["transcription"] is True
            assert data["services"]["llm"] is True
            assert data["services"]["tts"] is True
            assert data["services"]["animation"] is True
            assert data["cv_loaded"] is True
    
    @pytest.mark.unit
    def test_health_check_services_not_ready(self, fastapi_test_client):
        with patch('backend.main.cv_service') as mock_cv, \
             patch('backend.main.transcription_service') as mock_trans, \
             patch('backend.main.llm_service') as mock_llm, \
             patch('backend.main.tts_service') as mock_tts, \
             patch('backend.main.animation_service') as mock_anim:
            
            mock_cv.is_ready.return_value = False
            mock_trans.is_ready.return_value = False
            mock_llm.is_ready.return_value = False
            mock_tts.is_ready.return_value = False
            mock_anim.is_ready.return_value = False
            
            response = fastapi_test_client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["services"]["cv"] is False
            assert data["cv_loaded"] is False
    
    @pytest.mark.unit
    def test_get_cv_info_success(self, fastapi_test_client):
        with patch('backend.main.cv_service') as mock_cv:
            mock_cv.is_ready.return_value = True
            mock_cv.get_cv_info.return_value = {
                "has_cv": True,
                "default_name": "Henrique Lobato",
                "content_length": 1500
            }
            mock_cv.get_system_prompt.return_value = "You are Henrique Lobato..."
            
            response = fastapi_test_client.get("/cv")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["cv_info"]["has_cv"] is True
            assert "system_prompt_preview" in data
    
    @pytest.mark.unit
    def test_get_cv_info_service_not_ready(self, fastapi_test_client):
        with patch('backend.main.cv_service') as mock_cv:
            mock_cv.is_ready.return_value = False
            
            response = fastapi_test_client.get("/cv")
            
            assert response.status_code == 503
            assert "CV service not ready" in response.json()["detail"]
    
    @pytest.mark.unit
    def test_get_cv_info_exception(self, fastapi_test_client):
        with patch('backend.main.cv_service') as mock_cv:
            mock_cv.is_ready.return_value = True
            mock_cv.get_cv_info.side_effect = Exception("Test error")
            
            response = fastapi_test_client.get("/cv")
            
            assert response.status_code == 500
            assert "Test error" in response.json()["detail"]
    
    @pytest.mark.unit
    def test_reload_cv_success(self, fastapi_test_client):
        with patch('backend.main.cv_service') as mock_cv:
            mock_cv.is_ready.return_value = True
            mock_cv.reload_cv = AsyncMock(return_value=True)
            mock_cv.get_cv_info.return_value = {
                "has_cv": True,
                "content_length": 1500
            }
            
            response = fastapi_test_client.post("/cv/reload")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "CV reloaded successfully" in data["message"]
    
    @pytest.mark.unit
    def test_reload_cv_failure(self, fastapi_test_client):
        with patch('backend.main.cv_service') as mock_cv:
            mock_cv.is_ready.return_value = True
            mock_cv.reload_cv = AsyncMock(return_value=False)
            
            response = fastapi_test_client.post("/cv/reload")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "error"
            assert "Failed to reload CV" in data["message"]
    
    @pytest.mark.unit
    def test_reload_cv_service_not_ready(self, fastapi_test_client):
        with patch('backend.main.cv_service') as mock_cv:
            mock_cv.is_ready.return_value = False
            
            response = fastapi_test_client.post("/cv/reload")
            
            assert response.status_code == 503
    
    @pytest.mark.unit
    def test_transcribe_audio_success(self, fastapi_test_client):
        with patch('backend.main.transcription_service') as mock_trans:
            mock_response = MagicMock()
            mock_response.text = "Hello, this is a test transcription"
            mock_response.confidence = 0.95
            mock_response.language = "en"
            mock_response.processing_time = 1.2
            mock_response.word_count = 6
            mock_response.duration = 2.5
            mock_response.words = None
            mock_response.success = True
            mock_response.error_message = None
            mock_trans.transcribe = AsyncMock(return_value=mock_response)
            
            audio_data = b"fake audio data"
            files = {"audio_file": ("test.wav", audio_data, "audio/wav")}
            data = {"language": "auto"}
            
            response = fastapi_test_client.post("/transcribe", files=files, data=data)
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["text"] == "Hello, this is a test transcription"
            assert response_data["confidence"] == 0.95
    
    @pytest.mark.unit
    def test_transcribe_audio_exception(self, fastapi_test_client):
        with patch('backend.main.transcription_service') as mock_trans:
            mock_trans.transcribe = AsyncMock(side_effect=Exception("Transcription failed"))
            
            audio_data = b"fake audio data"
            files = {"audio_file": ("test.wav", audio_data, "audio/wav")}
            
            response = fastapi_test_client.post("/transcribe", files=files)
            
            assert response.status_code == 500
            assert "Transcription failed" in response.json()["detail"]
    
    @pytest.mark.unit
    def test_generate_response_success(self, fastapi_test_client):
        with patch('backend.main.llm_service') as mock_llm:
            mock_response = MagicMock()
            mock_response.response = "I have 8+ years of Python experience..."
            mock_response.model = "llama3"
            mock_response.conversation_id = "test-123"
            mock_response.processing_time = 2.1
            mock_response.success = True
            mock_response.error_message = None
            mock_response.timestamp = "2024-01-15T10:30:00"
            mock_response.token_count = 50
            mock_llm.generate_response = AsyncMock(return_value=mock_response)
            
            request_data = {
                "message": "What is your Python experience?",
                "conversation_id": "test-123"
            }
            
            response = fastapi_test_client.post("/respond", json=request_data)
            
            assert response.status_code == 200
            response_data = response.json()
            assert "Python experience" in response_data["response"]
            assert response_data["model"] == "llama3"
            assert response_data["success"] is True
    
    @pytest.mark.unit
    def test_generate_response_exception(self, fastapi_test_client):
        with patch('backend.main.llm_service') as mock_llm:
            mock_llm.generate_response = AsyncMock(side_effect=Exception("LLM error"))
            
            request_data = {
                "message": "What is your Python experience?"
            }
            
            response = fastapi_test_client.post("/respond", json=request_data)
            
            assert response.status_code == 500
            assert "LLM error" in response.json()["detail"]
    
    @pytest.mark.unit
    def test_synthesize_speech_success(self, fastapi_test_client):
        with patch('backend.main.tts_service') as mock_tts:
            mock_response = MagicMock()
            mock_response.audio_data = b"fake audio data"
            mock_response.audio_url = "/audio/test.wav"
            mock_response.duration = 3.5
            mock_response.sample_rate = 22050
            mock_response.processing_time = 1.8
            mock_response.voice_used = "default"
            mock_response.audio_format = "wav"
            mock_response.success = True
            mock_response.timestamp = "2024-01-15T10:30:00"
            mock_response.error_message = None
            mock_tts.synthesize = AsyncMock(return_value=mock_response)
            
            request_data = {
                "text": "Hello, this is a test",
                "voice_id": "default",
                "speed": 1.0,
                "pitch": 1.0
            }
            
            response = fastapi_test_client.post("/speak", json=request_data)
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["audio_url"] == "/audio/test.wav"
            assert response_data["duration"] == 3.5
    
    @pytest.mark.unit
    def test_synthesize_speech_exception(self, fastapi_test_client):
        with patch('backend.main.tts_service') as mock_tts:
            mock_tts.synthesize.side_effect = Exception("TTS error")
            
            request_data = {
                "text": "Hello, this is a test"
            }
            
            response = fastapi_test_client.post("/speak", json=request_data)
            
            assert response.status_code == 500
            assert "TTS error" in response.json()["detail"]
    
    @pytest.mark.unit
    def test_animate_avatar_success(self, fastapi_test_client):
        with patch('backend.main.animation_service') as mock_anim:
            mock_response = MagicMock()
            mock_response.video_data = b"fake video data"
            mock_response.video_url = "/video/avatar_animation.mp4"
            mock_response.duration = 5.2
            mock_response.fps = 25
            mock_response.resolution = "512x512"
            mock_response.processing_time = 10.5
            mock_response.file_size = 1024000
            mock_response.success = True
            mock_response.timestamp = "2024-01-15T10:30:00"
            mock_response.error_message = None
            mock_anim.animate = AsyncMock(return_value=mock_response)
            
            audio_data = b"fake audio data"
            files = {"audio_file": ("test.wav", audio_data, "audio/wav")}
            data = {"use_default_avatar": "true"}
            
            response = fastapi_test_client.post("/animate", files=files, data=data)
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["video_url"] == "/video/avatar_animation.mp4"
            assert response_data["duration"] == 5.2
    
    @pytest.mark.unit
    def test_animate_avatar_with_custom_image(self, fastapi_test_client):
        with patch('backend.main.animation_service') as mock_anim:
            mock_response = MagicMock()
            mock_response.video_url = "/video/custom_avatar.mp4"
            mock_response.duration = 4.8
            mock_response.fps = 25
            mock_response.resolution = "512x512"
            mock_response.processing_time = 8.3
            mock_response.file_size = 2048000
            mock_response.success = True
            mock_response.timestamp = "2024-01-15T10:30:00"
            mock_response.error_message = None
            mock_anim.animate = AsyncMock(return_value=mock_response)
            
            audio_data = b"fake audio data"
            image_data = b"fake image data"
            files = {
                "audio_file": ("test.wav", audio_data, "audio/wav"),
                "image_file": ("avatar.jpg", image_data, "image/jpeg")
            }
            data = {"use_default_avatar": "false"}
            
            response = fastapi_test_client.post("/animate", files=files, data=data)
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["video_url"] == "/video/custom_avatar.mp4"
    
    @pytest.mark.unit
    def test_animate_avatar_exception(self, fastapi_test_client):
        with patch('backend.main.animation_service') as mock_anim:
            mock_anim.animate.side_effect = Exception("Animation error")
            
            audio_data = b"fake audio data"
            files = {"audio_file": ("test.wav", audio_data, "audio/wav")}
            
            response = fastapi_test_client.post("/animate", files=files)
            
            assert response.status_code == 500
            assert "Animation error" in response.json()["detail"]
    
    @pytest.mark.unit
    def test_get_models_status(self, fastapi_test_client):
        with patch('backend.main.cv_service') as mock_cv, \
             patch('backend.main.transcription_service') as mock_trans, \
             patch('backend.main.llm_service') as mock_llm, \
             patch('backend.main.tts_service') as mock_tts, \
             patch('backend.main.animation_service') as mock_anim:
            
            mock_cv.is_ready.return_value = True
            mock_cv.has_cv.return_value = True
            mock_cv.cv_file_path = "data/cv.txt"
            
            mock_trans.is_ready.return_value = True
            mock_llm.is_ready.return_value = True
            mock_tts.is_ready.return_value = True
            mock_anim.is_ready.return_value = True
            
            response = fastapi_test_client.get("/models/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["cv"]["loaded"] is True
            assert data["cv"]["has_content"] is True
            assert data["whisper"]["loaded"] is True
            assert data["ollama"]["loaded"] is True
            assert data["ollama"]["cv_integration"] is True
            assert data["xtts"]["loaded"] is True
            assert data["sadtalker"]["loaded"] is True
    
    @pytest.mark.unit
    def test_get_websocket_stats(self, fastapi_test_client):
        with patch('backend.main.websocket_manager') as mock_ws:
            mock_ws.get_stats.return_value = {
                "active_connections": 3,
                "total_connections": 15,
                "messages_sent": 150,
                "messages_received": 120
            }
            
            response = fastapi_test_client.get("/ws/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["active_connections"] == 3
            assert data["total_connections"] == 15


class TestWebSocketHandling:
    
    @pytest.mark.unit
    async def test_handle_text_message_ping(self):
        from backend.main import handle_text_message
        
        mock_websocket = AsyncMock()
        connection_id = "test-123"
        data = {"type": "ping"}
        
        await handle_text_message(mock_websocket, connection_id, data)
        
        mock_websocket.send_json.assert_called_once()
        sent_data = mock_websocket.send_json.call_args[0][0]
        assert sent_data["type"] == "pong"
        assert "timestamp" in sent_data
    
    @pytest.mark.unit
    async def test_handle_text_message_clear_context(self):
        from backend.main import handle_text_message
        
        mock_websocket = AsyncMock()
        connection_id = "test-123"
        data = {"type": "clear_context"}
        
        with patch('backend.main.websocket_manager') as mock_ws_manager:
            await handle_text_message(mock_websocket, connection_id, data)
            
            mock_ws_manager.clear_conversation_context.assert_called_once_with(connection_id)
            mock_websocket.send_json.assert_called_once_with({"type": "context_cleared"})
    
    @pytest.mark.unit
    async def test_handle_text_message_get_context(self):
        from backend.main import handle_text_message
        
        mock_websocket = AsyncMock()
        connection_id = "test-123"
        data = {"type": "get_context"}
        
        with patch('backend.main.websocket_manager') as mock_ws_manager:
            mock_ws_manager.get_conversation_context.return_value = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
            
            await handle_text_message(mock_websocket, connection_id, data)
            
            mock_websocket.send_json.assert_called_once()
            sent_data = mock_websocket.send_json.call_args[0][0]
            assert sent_data["type"] == "conversation_context"
            assert len(sent_data["context"]) == 2
    
    @pytest.mark.unit
    async def test_handle_text_message_text_input(self):
        from backend.main import handle_text_message
        
        mock_websocket = AsyncMock()
        connection_id = "test-123"
        data = {"type": "text_input", "text": "What is your Python experience?"}
        
        with patch('backend.main.process_text_pipeline') as mock_pipeline:
            await handle_text_message(mock_websocket, connection_id, data)
            
            mock_pipeline.assert_called_once_with(
                mock_websocket, connection_id, "What is your Python experience?"
            )
    
    @pytest.mark.unit
    async def test_handle_text_message_settings(self):
        from backend.main import handle_text_message
        
        mock_websocket = AsyncMock()
        connection_id = "test-123"
        data = {
            "type": "settings", 
            "settings": {"voice_speed": 1.2, "model": "llama3"}
        }
        
        with patch('backend.main.update_user_settings') as mock_update:
            await handle_text_message(mock_websocket, connection_id, data)
            
            mock_update.assert_called_once_with(connection_id, {"voice_speed": 1.2, "model": "llama3"})
            mock_websocket.send_json.assert_called_once_with({"type": "settings_updated"})
    
    @pytest.mark.unit
    async def test_handle_text_message_unknown_type(self):
        from backend.main import handle_text_message
        
        mock_websocket = AsyncMock()
        connection_id = "test-123"
        data = {"type": "unknown_message_type"}
        
        await handle_text_message(mock_websocket, connection_id, data)
        
        mock_websocket.send_json.assert_called_once()
        sent_data = mock_websocket.send_json.call_args[0][0]
        assert sent_data["type"] == "error"
        assert "Unknown message type" in sent_data["message"]
    
    @pytest.mark.unit
    async def test_process_audio_pipeline_success(self):
        from backend.main import process_audio_pipeline
        
        mock_websocket = AsyncMock()
        connection_id = "test-123"
        audio_data = b"fake audio data"
        
        with patch('backend.main.transcription_service') as mock_trans, \
             patch('backend.main.websocket_manager') as mock_ws_manager, \
             patch('backend.main.process_text_pipeline') as mock_text_pipeline:
            
            mock_trans_response = MagicMock()
            mock_trans_response.text = "Hello, what is your Python experience?"
            mock_trans_response.confidence = 0.95
            mock_trans.transcribe = AsyncMock(return_value=mock_trans_response)
            
            await process_audio_pipeline(mock_websocket, connection_id, audio_data)
            
            mock_websocket.send_json.assert_called()
            sent_calls = mock_websocket.send_json.call_args_list
            transcription_call = sent_calls[0][0][0]
            assert transcription_call["type"] == "transcription"
            assert transcription_call["text"] == "Hello, what is your Python experience?"
            
            mock_ws_manager.add_conversation_message.assert_called_with(
                connection_id, "user", "Hello, what is your Python experience?"
            )
            
            mock_text_pipeline.assert_called_once()
    
    @pytest.mark.unit
    async def test_process_audio_pipeline_empty_transcription(self):
        from backend.main import process_audio_pipeline
        
        mock_websocket = AsyncMock()
        connection_id = "test-123"
        audio_data = b"fake audio data"
        
        with patch('backend.main.transcription_service') as mock_trans, \
             patch('backend.main.process_text_pipeline') as mock_text_pipeline:
            
            mock_trans_response = MagicMock()
            mock_trans_response.text = ""
            mock_trans_response.confidence = 0.1
            mock_trans.transcribe = AsyncMock(return_value=mock_trans_response)
            
            await process_audio_pipeline(mock_websocket, connection_id, audio_data)
            
            mock_text_pipeline.assert_not_called()
    
    @pytest.mark.unit
    async def test_process_audio_pipeline_exception(self):
        from backend.main import process_audio_pipeline
        
        mock_websocket = AsyncMock()
        connection_id = "test-123"
        audio_data = b"fake audio data"
        
        with patch('backend.main.transcription_service') as mock_trans:
            mock_trans.transcribe = AsyncMock(side_effect=Exception("Transcription failed"))
            
            await process_audio_pipeline(mock_websocket, connection_id, audio_data)
            
            mock_websocket.send_json.assert_called()
            error_call = mock_websocket.send_json.call_args[0][0]
            assert error_call["type"] == "error"
            assert "Audio processing failed" in error_call["message"]
    
    @pytest.mark.unit
    async def test_process_text_pipeline_success(self):
        from backend.main import process_text_pipeline
        
        mock_websocket = AsyncMock()
        connection_id = "test-123"
        user_text = "What is your Python experience?"
        
        with patch('backend.main.websocket_manager') as mock_ws_manager, \
             patch('backend.main.llm_service') as mock_llm, \
             patch('backend.main.tts_service') as mock_tts, \
             patch('backend.main.animation_service') as mock_anim:
            
            mock_ws_manager.get_conversation_context.return_value = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
            
            mock_llm_response = MagicMock()
            mock_llm_response.response = "I have 8+ years of Python experience..."
            mock_llm.generate_response = AsyncMock(return_value=mock_llm_response)
            
            mock_tts_response = MagicMock()
            mock_tts_response.audio_data = b"fake audio"
            mock_tts_response.audio_url = "/audio/response.wav"
            mock_tts.synthesize = AsyncMock(return_value=mock_tts_response)
            
            mock_anim_response = MagicMock()
            mock_anim_response.video_url = "/video/avatar.mp4"
            mock_anim_response.duration = 5.2
            mock_anim.animate = AsyncMock(return_value=mock_anim_response)
            
            await process_text_pipeline(mock_websocket, connection_id, user_text)
            
            mock_llm.generate_response.assert_called_once()
            mock_tts.synthesize.assert_called_once()
            mock_anim.animate.assert_called_once()
            
            assert mock_ws_manager.add_conversation_message.call_count == 1
            
            mock_websocket.send_json.assert_called()
            complete_call = None
            for call in mock_websocket.send_json.call_args_list:
                if call[0][0].get("type") == "complete_response":
                    complete_call = call[0][0]
                    break
            
            assert complete_call is not None
            assert complete_call["transcription"] == user_text
            assert "Python experience" in complete_call["llm_response"]
            assert complete_call["audio_url"] == "/audio/response.wav"
            assert complete_call["video_url"] == "/video/avatar.mp4"
    
    @pytest.mark.unit
    async def test_process_text_pipeline_exception(self):
        from backend.main import process_text_pipeline
        
        mock_websocket = AsyncMock()
        connection_id = "test-123"
        user_text = "What is your Python experience?"
        
        with patch('backend.main.websocket_manager') as mock_ws_manager, \
             patch('backend.main.llm_service') as mock_llm:
            
            mock_ws_manager.get_conversation_context.return_value = []
            mock_llm.generate_response = AsyncMock(side_effect=Exception("LLM failed"))
            
            await process_text_pipeline(mock_websocket, connection_id, user_text)
            
            mock_websocket.send_json.assert_called()
            error_call = None
            for call in mock_websocket.send_json.call_args_list:
                if call[0][0].get("type") == "error":
                    error_call = call[0][0]
                    break
            
            assert error_call is not None
            assert "Text processing failed" in error_call["message"]


class TestStartupShutdown:
    
    @pytest.mark.unit
    async def test_startup_event_success(self):
        from backend.main import startup_event
        
        with patch('backend.main.cv_service') as mock_cv, \
             patch('backend.main.transcription_service') as mock_trans, \
             patch('backend.main.llm_service') as mock_llm, \
             patch('backend.main.tts_service') as mock_tts, \
             patch('backend.main.animation_service') as mock_anim:
            
            mock_cv.initialize = AsyncMock()
            mock_cv.has_cv.return_value = True
            mock_cv.default_name = "Henrique Lobato"
            
            mock_trans.initialize = AsyncMock()
            mock_llm.initialize = AsyncMock()
            mock_tts.initialize = AsyncMock()
            mock_anim.initialize = AsyncMock()
            
            await startup_event()
            
            mock_cv.initialize.assert_called_once()
            mock_trans.initialize.assert_called_once()
            mock_llm.initialize.assert_called_once()
            mock_tts.initialize.assert_called_once()
            mock_anim.initialize.assert_called_once()
    
    @pytest.mark.unit
    async def test_startup_event_failure(self):
        from backend.main import startup_event
        
        with patch('backend.main.cv_service') as mock_cv:
            mock_cv.initialize = AsyncMock(side_effect=Exception("Initialization failed"))
            
            with pytest.raises(Exception) as exc_info:
                await startup_event()
            
            assert "Initialization failed" in str(exc_info.value)
    
    @pytest.mark.unit
    async def test_shutdown_event(self):
        from backend.main import shutdown_event
        
        with patch('backend.main.cv_service') as mock_cv, \
             patch('backend.main.transcription_service') as mock_trans, \
             patch('backend.main.llm_service') as mock_llm, \
             patch('backend.main.tts_service') as mock_tts, \
             patch('backend.main.animation_service') as mock_anim:
            
            mock_cv.cleanup = AsyncMock()
            mock_trans.cleanup = AsyncMock()
            mock_llm.cleanup = AsyncMock()
            mock_tts.cleanup = AsyncMock()
            mock_anim.cleanup = AsyncMock()
            
            await shutdown_event()
            
            mock_cv.cleanup.assert_called_once()
            mock_trans.cleanup.assert_called_once()
            mock_llm.cleanup.assert_called_once()
            mock_tts.cleanup.assert_called_once()
            mock_anim.cleanup.assert_called_once() 