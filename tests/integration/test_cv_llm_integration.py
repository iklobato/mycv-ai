"""
Integration tests for CV and LLM service interaction.

Tests the integration between CV service and LLM service including:
- CV loading and LLM prompt injection
- End-to-end personality consistency
- System prompt generation with CV context
- Real file operations (using temporary files)
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from backend.services.cv_service import CVService
from backend.services.llm_service import LLMService
from backend.models.schemas import LLMRequest


class TestCVLLMIntegration:
    """Integration tests for CV and LLM services."""
    
    @pytest.fixture
    async def cv_service_with_file(self, sample_cv_content):
        """Create CV service with real temporary file."""
        cv_service = CVService()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_cv_content)
            temp_path = Path(f.name)
        
        # Set CV service to use temp file
        cv_service.cv_file_path = temp_path
        
        # Initialize
        await cv_service.initialize()
        
        yield cv_service
        
        # Cleanup
        await cv_service.cleanup()
        if temp_path.exists():
            temp_path.unlink()
    
    @pytest.fixture
    def llm_service_with_mocked_ollama(self):
        """Create LLM service with mocked Ollama client."""
        llm_service = LLMService()
        
        # Mock the client
        mock_client = AsyncMock()
        llm_service.client = mock_client
        llm_service.is_initialized = True
        llm_service.available_models = ["llama3", "mistral"]
        
        return llm_service, mock_client
    
    @pytest.mark.integration
    async def test_cv_loads_and_enhances_llm_prompts(self, cv_service_with_file, sample_cv_content):
        """Test that CV content is loaded and enhances LLM prompts."""
        # Verify CV is loaded
        assert cv_service_with_file.has_cv() is True
        assert cv_service_with_file.get_cv_content() == sample_cv_content
        
        # Test system prompt generation
        system_prompt = cv_service_with_file.get_system_prompt()
        
        # Verify CV content is in prompt
        assert "=== CV START ===" in system_prompt
        assert "=== CV END ===" in system_prompt
        assert sample_cv_content in system_prompt
        assert "Henrique Lobato" in system_prompt or "HENRIQUE LOBATO" in system_prompt
        assert "respond as this person" in system_prompt.lower()
    
    @pytest.mark.integration
    async def test_llm_service_uses_cv_context(self, cv_service_with_file):
        """Test that LLM service automatically uses CV context."""
        llm_service, mock_client = self.llm_service_with_mocked_ollama()
        
        # Set CV service in LLM service
        llm_service.cv_service = cv_service_with_file
        
        # Mock Ollama response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "I have 8+ years of Python experience, specializing in Django and FastAPI..."}
        }
        mock_client.post.return_value = mock_response
        
        # Create request
        request = LLMRequest(
            message="What is your Python experience?",
            conversation_id="test-integration"
        )
        
        with patch('backend.services.llm_service.settings') as mock_settings:
            mock_settings.LLM_MODEL = "llama3"
            mock_settings.MODELS_CONFIG = {
                "ollama": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "num_ctx": 2048,
                    "num_predict": 150
                }
            }
            
            # Generate response
            response = await llm_service.generate_response(request)
        
        # Verify response was generated
        assert response.success is True
        assert "Python experience" in response.response
        
        # Verify Ollama was called with CV-enhanced prompt
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        
        # Check that messages include system prompt with CV
        messages = payload["messages"]
        system_message = next((msg for msg in messages if msg["role"] == "system"), None)
        assert system_message is not None
        assert "=== CV START ===" in system_message["content"]
        assert "=== CV END ===" in system_message["content"]
        assert "Henrique Lobato" in system_message["content"] or "HENRIQUE LOBATO" in system_message["content"]
    
    @pytest.mark.integration
    async def test_cv_service_fallback_when_file_missing(self):
        """Test LLM service behavior when CV file is missing."""
        cv_service = CVService()
        cv_service.cv_file_path = Path("nonexistent_file.txt")
        
        # Initialize CV service (should not fail)
        await cv_service.initialize()
        
        # Verify no CV is loaded
        assert cv_service.has_cv() is False
        
        # Create LLM service
        llm_service, mock_client = self.llm_service_with_mocked_ollama()
        llm_service.cv_service = cv_service
        
        # Mock Ollama response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "I am a Python developer with general experience..."}
        }
        mock_client.post.return_value = mock_response
        
        # Create request
        request = LLMRequest(message="Tell me about yourself")
        
        with patch('backend.services.llm_service.settings') as mock_settings:
            mock_settings.LLM_MODEL = "llama3"
            mock_settings.MODELS_CONFIG = {"ollama": {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_ctx": 2048, "num_predict": 150}}
            
            response = await llm_service.generate_response(request)
        
        # Verify fallback behavior
        assert response.success is True
        
        # Check that system prompt uses fallback
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        messages = payload["messages"]
        system_message = next((msg for msg in messages if msg["role"] == "system"), None)
        assert system_message is not None
        assert "CV information is not currently available" in system_message["content"]
        assert "Henrique Lobato" in system_message["content"]
        
        await cv_service.cleanup()
    
    @pytest.mark.integration
    async def test_cv_reload_updates_llm_prompts(self, sample_cv_content):
        """Test that CV reload updates LLM prompt generation."""
        cv_service = CVService()
        
        # Create temporary file with initial content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_cv_content)
            temp_path = Path(f.name)
        
        cv_service.cv_file_path = temp_path
        await cv_service.initialize()
        
        # Get initial system prompt
        initial_prompt = cv_service.get_system_prompt()
        assert "Henrique Lobato" in initial_prompt
        
        # Update CV file with different content
        updated_content = """JANE SMITH
Lead Data Scientist

SUMMARY
Experienced Data Scientist with expertise in machine learning and AI.

TECHNICAL SKILLS
Programming Languages:
- Python (Expert) - 10+ years
- R (Advanced) - 5+ years
- SQL (Advanced)

PROFESSIONAL EXPERIENCE
Lead Data Scientist | TechCorp | 2020 - Present
- Developed ML models for predictive analytics
- Led team of 5 data scientists"""
        
        with open(temp_path, 'w') as f:
            f.write(updated_content)
        
        # Reload CV
        success = await cv_service.reload_cv()
        assert success is True
        
        # Get updated system prompt
        updated_prompt = cv_service.get_system_prompt()
        assert "JANE SMITH" in updated_prompt
        assert "Lead Data Scientist" in updated_prompt
        assert updated_content in updated_prompt
        
        # Verify old content is not in new prompt
        assert "Henrique Lobato" not in updated_prompt
        
        # Cleanup
        await cv_service.cleanup()
        if temp_path.exists():
            temp_path.unlink()
    
    @pytest.mark.integration
    async def test_conversation_history_with_cv_context(self, cv_service_with_file):
        """Test conversation history management with CV context."""
        llm_service, mock_client = self.llm_service_with_mocked_ollama()
        llm_service.cv_service = cv_service_with_file
        
        conversation_id = "test-conversation"
        
        # Mock Ollama responses
        responses = [
            {"message": {"content": "I'm Henrique, a Senior Python Developer with 8+ years of experience."}},
            {"message": {"content": "Yes, I've worked extensively with Django, FastAPI, and Flask."}},
            {"message": {"content": "My most recent project is an AI Avatar system using real-time communication."}}
        ]
        
        mock_client.post.side_effect = [
            MagicMock(status_code=200, json=lambda: resp) for resp in responses
        ]
        
        # Simulate conversation
        questions = [
            "Who are you?",
            "Do you have experience with web frameworks?",
            "What's your most recent project?"
        ]
        
        with patch('backend.services.llm_service.settings') as mock_settings:
            mock_settings.LLM_MODEL = "llama3"
            mock_settings.MODELS_CONFIG = {"ollama": {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_ctx": 2048, "num_predict": 150}}
            
            for i, question in enumerate(questions):
                request = LLMRequest(
                    message=question,
                    conversation_id=conversation_id
                )
                
                response = await llm_service.generate_response(request)
                assert response.success is True
                assert response.conversation_id == conversation_id
        
        # Verify conversation history is maintained
        history = llm_service._get_conversation_history(conversation_id)
        assert len(history) == 6  # 3 questions + 3 responses
        
        # Verify each Ollama call included CV context
        assert mock_client.post.call_count == 3
        for call_args in mock_client.post.call_args_list:
            payload = call_args[1]["json"]
            messages = payload["messages"]
            
            # Should have system message with CV
            system_message = next((msg for msg in messages if msg["role"] == "system"), None)
            assert system_message is not None
            assert "=== CV START ===" in system_message["content"]
    
    @pytest.mark.integration
    async def test_cv_stats_and_info_accuracy(self, cv_service_with_file, sample_cv_content):
        """Test accuracy of CV statistics and info."""
        # Get CV info
        cv_info = cv_service_with_file.get_cv_info()
        
        assert cv_info["has_cv"] is True
        assert cv_info["content_length"] == len(sample_cv_content)
        assert cv_info["default_name"] == "Henrique Lobato"
        assert "Python Developer" in cv_info["default_title"]
        assert cv_info["file_exists"] is True
        assert cv_info["last_loaded"] is not None
        
        # Get CV stats
        cv_stats = cv_service_with_file.get_cv_stats()
        
        expected_char_count = len(sample_cv_content)
        expected_word_count = len(sample_cv_content.split())
        expected_line_count = len([line for line in sample_cv_content.split('\n') if line.strip()])
        expected_paragraph_count = len([p for p in sample_cv_content.split('\n\n') if p.strip()])
        
        assert cv_stats["character_count"] == expected_char_count
        assert cv_stats["word_count"] == expected_word_count
        assert cv_stats["line_count"] == expected_line_count
        assert cv_stats["paragraph_count"] == expected_paragraph_count
    
    @pytest.mark.integration
    async def test_end_to_end_cv_llm_workflow(self, sample_cv_content):
        """Test complete end-to-end workflow from CV loading to LLM response."""
        # Step 1: Create and initialize CV service
        cv_service = CVService()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_cv_content)
            temp_path = Path(f.name)
        
        cv_service.cv_file_path = temp_path
        await cv_service.initialize()
        
        # Step 2: Verify CV is properly loaded
        assert cv_service.is_ready() is True
        assert cv_service.has_cv() is True
        
        # Step 3: Create and setup LLM service
        llm_service, mock_client = self.llm_service_with_mocked_ollama()
        llm_service.cv_service = cv_service
        
        # Step 4: Mock realistic response based on CV
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": "I'm Henrique Lobato, a Senior Python Developer & AI Specialist. I have over 8 years of Python experience and currently work at AI Avatar Systems where I develop innovative AI-powered applications. My expertise includes Django, FastAPI, Flask, and I've recently built an AI Avatar Video Call System using technologies like Whisper, Ollama, XTTS-v2, and SadTalker."
            }
        }
        mock_client.post.return_value = mock_response
        
        # Step 5: Generate response that should reflect CV personality
        request = LLMRequest(
            message="Please introduce yourself and tell me about your background",
            conversation_id="end-to-end-test"
        )
        
        with patch('backend.services.llm_service.settings') as mock_settings:
            mock_settings.LLM_MODEL = "llama3"
            mock_settings.MODELS_CONFIG = {"ollama": {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_ctx": 2048, "num_predict": 150}}
            
            response = await llm_service.generate_response(request)
        
        # Step 6: Verify the complete workflow
        assert response.success is True
        assert "Henrique Lobato" in response.response
        assert "Python" in response.response
        assert response.conversation_id == "end-to-end-test"
        
        # Step 7: Verify the system prompt was properly constructed
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        messages = payload["messages"]
        
        # Find system message
        system_message = next((msg for msg in messages if msg["role"] == "system"), None)
        assert system_message is not None
        
        system_content = system_message["content"]
        assert "You are Henrique Lobato" in system_content
        assert "=== CV START ===" in system_content
        assert sample_cv_content in system_content
        assert "=== CV END ===" in system_content
        assert "respond as this person" in system_content.lower()
        
        # Step 8: Cleanup
        await cv_service.cleanup()
        await llm_service.cleanup()
        if temp_path.exists():
            temp_path.unlink()
    
    def llm_service_with_mocked_ollama(self):
        """Helper method to create LLM service with mocked Ollama."""
        llm_service = LLMService()
        
        # Mock the client
        mock_client = AsyncMock()
        llm_service.client = mock_client
        llm_service.is_initialized = True
        llm_service.available_models = ["llama3", "mistral"]
        
        return llm_service, mock_client 