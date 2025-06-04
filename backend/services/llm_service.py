"""
LLM Service using Ollama

Handles AI text generation with support for:
- Local LLM hosting via Ollama
- Multiple model support (Llama3, Mistral, Phi-3)
- Conversation context management
- Streaming responses
- CV-based personality prompts
"""

import asyncio
import logging
import time
import uuid
from typing import Optional, Dict, Any, List, AsyncIterable
import json

import httpx
from httpx import AsyncClient

from ..models.schemas import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

# Try to import settings, but use fallbacks if not available (for testing)
try:
    from ..config import settings
except ImportError:
    # Fallback settings for testing
    class MockSettings:
        OLLAMA_BASE_URL = "http://localhost:11434"
        LLM_MODEL = "llama3"
        LLM_TIMEOUT = 30
        MODELS_CONFIG = {
            "ollama": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "num_ctx": 2048,
                "num_predict": 150
            }
        }
    settings = MockSettings()


class LLMService:
    """Service for AI text generation using Ollama with CV-based personality."""
    
    def __init__(self):
        self.client: Optional[AsyncClient] = None
        self.is_initialized = False
        self.available_models = []
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        self.cv_service = None  # Will be set during initialization
        
    async def initialize(self):
        """Initialize the Ollama client and check model availability."""
        logger.info("Initializing LLM service...")
        
        try:
            # Import CV service here to avoid circular imports
            from .cv_service import cv_service
            self.cv_service = cv_service
            
            # Create HTTP client
            self.client = AsyncClient(
                base_url=settings.OLLAMA_BASE_URL,
                timeout=httpx.Timeout(settings.LLM_TIMEOUT),
                headers={"Content-Type": "application/json"}
            )
            
            # Check Ollama server connectivity
            await self._check_ollama_health()
            
            # Get available models
            await self._get_available_models()
            
            # Ensure the configured model is available
            await self._ensure_model_available()
            
            self.is_initialized = True
            logger.info("LLM service initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise
    
    async def _check_ollama_health(self):
        """Check if Ollama server is running."""
        try:
            response = await self.client.get("/api/tags")
            if response.status_code != 200:
                raise Exception(f"Ollama server not healthy: {response.status_code}")
            
            logger.info("Ollama server is healthy")
            
        except httpx.ConnectError:
            raise Exception(
                f"Cannot connect to Ollama server at {settings.OLLAMA_BASE_URL}. "
                "Make sure Ollama is running: 'ollama serve'"
            )
        except Exception as e:
            raise Exception(f"Ollama health check failed: {e}")
    
    async def _get_available_models(self):
        """Get list of available models from Ollama."""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            
            data = response.json()
            self.available_models = [model["name"] for model in data.get("models", [])]
            
            logger.info(f"Available models: {self.available_models}")
            
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            self.available_models = []
    
    async def _ensure_model_available(self):
        """Ensure the configured model is available, pull if necessary."""
        model_name = settings.LLM_MODEL
        
        if model_name not in self.available_models:
            logger.info(f"Model {model_name} not found locally, attempting to pull...")
            
            try:
                await self._pull_model(model_name)
                # Refresh available models
                await self._get_available_models()
                
                if model_name not in self.available_models:
                    raise Exception(f"Failed to pull model {model_name}")
                    
            except Exception as e:
                logger.error(f"Failed to pull model {model_name}: {e}")
                
                # Try to use any available model as fallback
                if self.available_models:
                    fallback_model = self.available_models[0]
                    logger.warning(f"Using fallback model: {fallback_model}")
                    settings.LLM_MODEL = fallback_model
                else:
                    raise Exception("No models available and failed to pull configured model")
    
    async def _pull_model(self, model_name: str):
        """Pull a model from Ollama registry."""
        logger.info(f"Pulling model: {model_name}")
        
        payload = {"name": model_name}
        
        async with self.client.stream(
            "POST", 
            "/api/pull", 
            json=payload
        ) as response:
            if response.status_code != 200:
                raise Exception(f"Failed to pull model: {response.status_code}")
            
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        status = data.get("status", "")
                        
                        if "progress" in data:
                            logger.info(f"Pull progress: {status}")
                        elif status == "success":
                            logger.info(f"Successfully pulled model: {model_name}")
                            break
                            
                    except json.JSONDecodeError:
                        continue
    
    def _get_cv_enhanced_system_prompt(self, custom_instructions: Optional[str] = None) -> str:
        """Get the CV-enhanced system prompt for AI avatar personality."""
        if self.cv_service and self.cv_service.is_ready():
            return self.cv_service.get_system_prompt(custom_instructions)
        else:
            # Fallback system prompt when CV service is not available
            fallback_prompt = """You are Henrique Lobato, a Senior Python Developer. You are having a natural conversation with someone.

You are knowledgeable about Python development, AI systems, and web applications. Keep your responses conversational and concise for voice interaction.

Since your detailed CV information is not currently available, draw on general knowledge about Python development and AI while being engaging and helpful."""
            
            if custom_instructions:
                fallback_prompt += f"\n\nAdditional instructions: {custom_instructions}"
            
            return fallback_prompt
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate AI response using Ollama with CV-based personality.
        
        Args:
            request: LLM request with user text and context
            
        Returns:
            LLMResponse with generated text
        """
        if not self.is_initialized:
            raise RuntimeError("LLM service not initialized")
        
        start_time = time.time()
        
        try:
            # Prepare conversation context
            conversation_id = request.conversation_id or str(uuid.uuid4())
            
            # Get conversation history
            conversation_history = self._get_conversation_history(conversation_id)
            
            # Add current user message
            conversation_history.append({"role": "user", "content": request.message})
            
            # Get CV-enhanced system prompt
            cv_system_prompt = self._get_cv_enhanced_system_prompt(request.system_prompt)
            
            # Prepare messages for Ollama
            messages = self._prepare_messages(
                conversation_history,
                cv_system_prompt
            )
            
            # Generate response
            response_text = await self._generate_ollama_response(request, messages)
            
            # Add AI response to conversation history
            conversation_history.append({"role": "assistant", "content": response_text})
            
            # Store updated conversation
            self._store_conversation(conversation_id, conversation_history)
            
            processing_time = time.time() - start_time
            
            # Create response
            response = LLMResponse(
                response=response_text,
                model=settings.LLM_MODEL,
                conversation_id=conversation_id,
                processing_time=processing_time
            )
            
            logger.debug(f"LLM response generated in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            processing_time = time.time() - start_time
            
            return LLMResponse(
                response="I apologize, but I'm having trouble generating a response right now. Could you please try again?",
                model=settings.LLM_MODEL,
                conversation_id=request.conversation_id,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a given conversation ID."""
        return self.conversations.get(conversation_id, [])
    
    def _store_conversation(self, conversation_id: str, history: List[Dict[str, str]]):
        """Store conversation history, keeping only recent messages."""
        # Keep only last 10 exchanges (20 messages) to manage context length
        max_messages = 20
        if len(history) > max_messages:
            # Keep system message if it exists, then recent messages
            system_messages = [msg for msg in history if msg["role"] == "system"]
            other_messages = [msg for msg in history if msg["role"] != "system"]
            
            if system_messages:
                recent_messages = system_messages[:1] + other_messages[-max_messages+1:]
            else:
                recent_messages = other_messages[-max_messages:]
            
            self.conversations[conversation_id] = recent_messages
        else:
            self.conversations[conversation_id] = history
    
    def _prepare_messages(
        self, 
        conversation_history: List[Dict[str, str]], 
        system_prompt: str
    ) -> List[Dict[str, str]]:
        """Prepare messages for Ollama API."""
        messages = []
        
        # Add CV-enhanced system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(conversation_history)
        
        return messages
    
    async def _generate_ollama_response(
        self, 
        request: LLMRequest, 
        messages: List[Dict[str, str]]
    ) -> str:
        """Generate response using Ollama API."""
        # Use fallback config if MODELS_CONFIG not available
        try:
            ollama_config = settings.MODELS_CONFIG["ollama"]
        except (AttributeError, KeyError):
            ollama_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "num_ctx": 2048,
                "num_predict": 150
            }
        
        payload = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": ollama_config["top_p"],
                "top_k": ollama_config["top_k"],
                "repeat_penalty": ollama_config["repeat_penalty"],
                "num_ctx": ollama_config["num_ctx"],
                "num_predict": request.max_tokens,
            }
        }
        
        try:
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"].strip()
            else:
                raise Exception("Invalid response format from Ollama")
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Ollama API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
    
    async def generate_streaming_response(
        self, 
        request: LLMRequest
    ) -> AsyncIterable[str]:
        """
        Generate streaming AI response with CV-based personality.
        
        Args:
            request: LLM request with user text and context
            
        Yields:
            Partial response chunks
        """
        if not self.is_initialized:
            raise RuntimeError("LLM service not initialized")
        
        try:
            # Prepare conversation context
            conversation_id = request.conversation_id or str(uuid.uuid4())
            conversation_history = self._get_conversation_history(conversation_id)
            conversation_history.append({"role": "user", "content": request.message})
            
            # Get CV-enhanced system prompt
            cv_system_prompt = self._get_cv_enhanced_system_prompt(request.system_prompt)
            
            messages = self._prepare_messages(
                conversation_history,
                cv_system_prompt
            )
            
            # Use fallback config if MODELS_CONFIG not available
            try:
                ollama_config = settings.MODELS_CONFIG["ollama"]
            except (AttributeError, KeyError):
                ollama_config = {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "num_ctx": 2048,
                    "num_predict": 150
                }
            
            payload = {
                "model": settings.LLM_MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": request.temperature,
                    "top_p": ollama_config["top_p"],
                    "top_k": ollama_config["top_k"],
                    "repeat_penalty": ollama_config["repeat_penalty"],
                    "num_ctx": ollama_config["num_ctx"],
                    "num_predict": request.max_tokens,
                }
            }
            
            full_response = ""
            
            async with self.client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code != 200:
                    raise Exception(f"Ollama streaming error: {response.status_code}")
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            
                            if "message" in data and "content" in data["message"]:
                                chunk = data["message"]["content"]
                                full_response += chunk
                                yield chunk
                            
                            # Check if stream is done
                            if data.get("done", False):
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
            # Store complete response in conversation history
            conversation_history.append({"role": "assistant", "content": full_response})
            self._store_conversation(conversation_id, conversation_history)
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error: {str(e)}"
    
    async def clear_conversation(self, conversation_id: str):
        """Clear conversation history for a specific conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation: {conversation_id}")
    
    async def get_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """Get a summary of the conversation."""
        if conversation_id not in self.conversations:
            return None
        
        history = self.conversations[conversation_id]
        
        if len(history) < 2:
            return "No significant conversation yet."
        
        # Create a summary request with CV context
        cv_system_prompt = self._get_cv_enhanced_system_prompt()
        summary_messages = [
            {"role": "system", "content": f"{cv_system_prompt}\n\nSummarize the following conversation in 1-2 sentences:"},
            {"role": "user", "content": str(history)}
        ]
        
        try:
            payload = {
                "model": settings.LLM_MODEL,
                "messages": summary_messages,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 100,
                }
            }
            
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return "Unable to generate summary."
    
    def is_ready(self) -> bool:
        """Check if the service is ready."""
        return self.is_initialized and self.client is not None
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up LLM service...")
        
        try:
            if self.client:
                await self.client.aclose()
            
            logger.info("LLM service cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during LLM service cleanup: {e}")
        finally:
            # Always reset state regardless of exceptions
            self.client = None
            self.conversations.clear()
            self.is_initialized = False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.is_initialized:
            return {"error": "Service not initialized"}
        
        try:
            response = await self.client.post(
                "/api/show",
                json={"name": settings.LLM_MODEL}
            )
            response.raise_for_status()
            
            model_info = response.json()
            
            # Add CV service information
            cv_info = None
            if self.cv_service and self.cv_service.is_ready():
                cv_info = {
                    "has_cv": self.cv_service.has_cv(),
                    "cv_loaded": self.cv_service.last_loaded.isoformat() if self.cv_service.last_loaded else None
                }
            
            model_info["cv_integration"] = cv_info
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    async def list_models(self) -> List[str]:
        """Get list of all available models."""
        return self.available_models.copy()
    
    def set_system_prompt(self, prompt: str):
        """Set a custom system prompt (note: CV context will still be included)."""
        logger.info("Custom system prompt will be combined with CV context")
    
    def get_conversation_count(self) -> int:
        """Get the number of active conversations."""
        return len(self.conversations) 