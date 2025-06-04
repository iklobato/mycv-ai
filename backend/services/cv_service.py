import asyncio
import logging
import os
from pathlib import Path
from typing import Optional
import aiofiles
from datetime import datetime

logger = logging.getLogger(__name__)


class CVService:
    
    def __init__(self, cv_file_path: Path = None):
        self.cv_file_path = cv_file_path or Path("data/cv.txt")
        self.cv_content: Optional[str] = None
        self.last_loaded: Optional[datetime] = None
        self.is_initialized = False
        
        self.default_name = "Henrique Lobato"
        self.default_title = "Senior Python Developer"
        
    async def initialize(self) -> bool:
        logger.info("Initializing CV service...")
        
        try:
            success = await self.load_cv()
            self.is_initialized = True
            
            if success:
                logger.info("CV service initialized successfully with CV content")
            else:
                logger.warning("CV service initialized without CV content")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to initialize CV service: {e}")
            self.is_initialized = False
            raise
    
    async def load_cv(self) -> bool:
        try:
            if not self.cv_file_path.exists():
                logger.warning(f"CV file not found: {self.cv_file_path}")
                return False
            
            async with aiofiles.open(self.cv_file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if not content.strip():
                logger.warning("CV file is empty")
                return False
            
            self.cv_content = content.strip()
            self.last_loaded = datetime.now()
            self._extract_basic_info()
            
            logger.info(f"CV loaded successfully: {len(self.cv_content)} characters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load CV: {e}")
            return False
    
    async def reload_cv(self) -> bool:
        logger.info("Reloading CV content...")
        return await self.load_cv()
    
    def has_cv(self) -> bool:
        return self.cv_content is not None and bool(self.cv_content.strip())
    
    def get_cv_content(self) -> Optional[str]:
        return self.cv_content
    
    def is_ready(self) -> bool:
        return self.is_initialized
    
    def _extract_basic_info(self):
        if not self.cv_content:
            return
            
        lines = [line.strip() for line in self.cv_content.split('\n') if line.strip()]
        if not lines:
            return
            
        # Extract name from first line if reasonable length
        first_line = lines[0]
        if len(first_line) <= 50:  # Reasonable name length
            self.default_name = first_line
        
        # Extract title from second line if it looks like a job title
        if len(lines) > 1:
            second_line = lines[1]
            # Simple heuristic: if it contains common job-related keywords
            job_keywords = ['developer', 'engineer', 'manager', 'director', 'analyst', 
                          'designer', 'consultant', 'specialist', 'architect', 'lead',
                          'senior', 'junior', 'principal', 'staff']
            if any(keyword.lower() in second_line.lower() for keyword in job_keywords):
                self.default_title = second_line
    
    def get_system_prompt(self, custom_instructions: Optional[str] = None) -> str:
        if not self.has_cv():
            base_prompt = f"""You are {self.default_name}, a {self.default_title}. You are having a natural conversation with someone.

You are knowledgeable about Python development, AI systems, and web applications. Keep your responses conversational and concise for voice interaction.

Since your detailed CV information is not currently available, draw on general knowledge about Python development and AI while being engaging and helpful."""
        else:
            base_prompt = f"""You are {self.default_name}. Here is your CV information:

=== CV START ===
{self.cv_content}
=== CV END ===

Based on this CV, respond as this person in a natural conversation. Keep responses conversational and appropriate for voice interaction."""
        
        if custom_instructions:
            base_prompt += f"\n\nAdditional instructions: {custom_instructions}"
        
        return base_prompt
    
    def get_cv_info(self) -> dict:
        name = self.default_name
        title = self.default_title
        
        if self.has_cv() and self.cv_content:
            lines = [line.strip() for line in self.cv_content.split('\n') if line.strip()]
            if lines:
                name = lines[0]
                if len(lines) > 1:
                    title = lines[1]
        
        content_preview = None
        if self.has_cv():
            preview_text = self.cv_content[:300] if len(self.cv_content) > 300 else self.cv_content
            content_preview = preview_text + "..." if len(self.cv_content) > 300 else preview_text
        
        return {
            "has_cv": self.has_cv(),
            "default_name": name,
            "default_title": title,
            "content_length": len(self.cv_content) if self.cv_content else 0,
            "last_loaded": self.last_loaded.isoformat() if self.last_loaded else None,
            "file_exists": self.cv_file_path.exists(),
            "content_preview": content_preview
        }
    
    def get_cv_stats(self) -> dict:
        if not self.has_cv():
            return {
                "character_count": 0,
                "word_count": 0,
                "line_count": 0,
                "paragraph_count": 0
            }
        
        content = self.cv_content
        
        character_count = len(content)
        word_count = len(content.split()) if content.strip() else 0
        line_count = len([line for line in content.split('\n') if line.strip()]) if content.strip() else 0
        
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            "character_count": character_count,
            "word_count": word_count,
            "line_count": line_count,
            "paragraph_count": paragraph_count
        }
    
    async def cleanup(self):
        logger.info("Cleaning up CV service...")
        self.cv_content = None
        self.last_loaded = None
        self.is_initialized = False


cv_service = CVService() 