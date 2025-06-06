"""
Unit tests for CV Service.

Tests all CV service functionality including:
- CV loading and validation
- System prompt generation
- Info extraction
- Error handling
- Edge cases
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from backend.services.cv_service import CVService


@pytest.fixture
def cv_service():
    return CVService()


class TestCVService:
    """Test cases for CVService class."""
    
    @pytest.mark.unit
    def test_cv_service_initialization(self, tmp_path):
        cv_file = tmp_path / "test_cv.txt"
        service = CVService(cv_file_path=cv_file)
        
        assert service.cv_file_path == cv_file
        assert service.cv_content is None
        assert service.last_loaded is None
        assert not service.is_initialized
        assert service.default_name == "Henrique Lobato"
        assert service.default_title == "Senior Python Developer"
    
    @pytest.mark.unit
    async def test_initialize_success(self, cv_service, sample_cv_content):
        with patch.object(cv_service, 'load_cv', return_value=True) as mock_load:
            result = await cv_service.initialize()
            
            assert result is True
            assert cv_service.is_initialized is True
            mock_load.assert_called_once()
    
    @pytest.mark.unit
    async def test_initialize_failure(self, cv_service):
        with patch.object(cv_service, 'load_cv', side_effect=Exception("Load failed")):
            with pytest.raises(Exception, match="Load failed"):
                await cv_service.initialize()
            
            assert cv_service.is_initialized is False
    
    @pytest.mark.unit
    async def test_initialize_no_cv_content(self, cv_service):
        """Test initialization when load_cv returns False (no CV content)."""
        with patch.object(cv_service, 'load_cv', return_value=False):
            result = await cv_service.initialize()
            
            assert result is False
            assert cv_service.is_initialized is True  # Still initialized, just without CV
    
    @pytest.mark.unit
    async def test_load_cv_success(self, cv_service, sample_cv_content):
        mock_file = AsyncMock()
        mock_file.__aenter__.return_value.read.return_value = sample_cv_content
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('aiofiles.open', return_value=mock_file):
            
            result = await cv_service.load_cv()
            
            assert result is True
            assert cv_service.cv_content == sample_cv_content
            assert cv_service.last_loaded is not None
            assert isinstance(cv_service.last_loaded, datetime)
    
    @pytest.mark.unit
    async def test_load_cv_file_not_found(self, cv_service):
        with patch('pathlib.Path.exists', return_value=False):
            result = await cv_service.load_cv()
            
            assert result is False
            assert cv_service.cv_content is None
            assert cv_service.last_loaded is None
    
    @pytest.mark.unit
    async def test_load_cv_content_too_short(self, cv_service):
        short_content = "   "  # Empty content with whitespace
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('aiofiles.open', mock_open(read_data=short_content)):
            
            result = await cv_service.load_cv()
            
            assert result is False
            assert cv_service.cv_content is None
    
    @pytest.mark.unit
    async def test_load_cv_exception(self, cv_service):
        with patch('pathlib.Path.exists', return_value=True), \
             patch('aiofiles.open', side_effect=Exception("File read error")):
            
            result = await cv_service.load_cv()
            
            assert result is False
            assert cv_service.cv_content is None
    
    @pytest.mark.unit
    def test_extract_basic_info_with_name_and_title(self, cv_service, sample_cv_content):
        cv_service.cv_content = sample_cv_content
        
        cv_service._extract_basic_info()
        
        assert cv_service.default_name == "Henrique Lobato"
        assert "Developer" in cv_service.default_title
    
    @pytest.mark.unit
    def test_extract_basic_info_no_content(self, cv_service):
        cv_service.cv_content = None
        original_name = cv_service.default_name
        original_title = cv_service.default_title
        
        cv_service._extract_basic_info()
        
        assert cv_service.default_name == original_name
        assert cv_service.default_title == original_title
    
    @pytest.mark.unit
    def test_extract_basic_info_simple_format(self, cv_service):
        simple_cv = """John Smith
Software Engineer
Experience with Python development."""
        cv_service.cv_content = simple_cv
        
        cv_service._extract_basic_info()
        
        assert cv_service.default_name == "John Smith"
        assert cv_service.default_title == "Software Engineer"
    
    @pytest.mark.unit
    def test_extract_basic_info_no_valid_title(self, cv_service):
        cv_without_title = """Jane Doe
Some random text without job title
More random content here."""
        cv_service.cv_content = cv_without_title
        original_title = cv_service.default_title
        
        cv_service._extract_basic_info()
        
        assert cv_service.default_name == "Jane Doe"
        assert cv_service.default_title == original_title
    
    @pytest.mark.unit
    def test_extract_basic_info_long_first_line(self, cv_service):
        cv_with_long_line = """This is a very long first line that should not be considered as a name because it exceeds the character limit
Software Developer
More content here."""
        cv_service.cv_content = cv_with_long_line
        original_name = cv_service.default_name
        
        cv_service._extract_basic_info()
        
        assert cv_service.default_name == original_name
    
    @pytest.mark.unit
    def test_extract_basic_info_empty_lines_only(self, cv_service):
        """Test _extract_basic_info with content that has no non-empty lines."""
        cv_with_empty_lines = "\n\n   \n\t\n   \n"
        cv_service.cv_content = cv_with_empty_lines
        original_name = cv_service.default_name
        original_title = cv_service.default_title
        
        cv_service._extract_basic_info()
        
        # Should not change defaults when there are no valid lines
        assert cv_service.default_name == original_name
        assert cv_service.default_title == original_title
    
    @pytest.mark.unit
    async def test_reload_cv(self, cv_service):
        with patch.object(cv_service, 'load_cv', return_value=True) as mock_load:
            result = await cv_service.reload_cv()
            
            assert result is True
            mock_load.assert_called_once()
    
    @pytest.mark.unit
    def test_get_system_prompt_with_cv(self, cv_service, sample_cv_content):
        cv_service.cv_content = sample_cv_content
        
        prompt = cv_service.get_system_prompt()
        
        assert "Henrique Lobato" in prompt
        assert sample_cv_content in prompt
        assert "natural conversation" in prompt
    
    @pytest.mark.unit
    def test_get_system_prompt_without_cv(self, cv_service):
        cv_service.cv_content = None
        
        prompt = cv_service.get_system_prompt()
        
        assert "Henrique Lobato" in prompt
        assert "Senior Python Developer" in prompt
        assert "not currently available" in prompt
    
    @pytest.mark.unit
    def test_get_system_prompt_with_custom_instructions(self, cv_service, sample_cv_content):
        cv_service.cv_content = sample_cv_content
        custom_instructions = "Be extra friendly and use emojis"
        
        prompt = cv_service.get_system_prompt(custom_instructions)
        
        assert custom_instructions in prompt
        assert "Additional instructions:" in prompt
    
    @pytest.mark.unit
    def test_has_cv_with_content(self, cv_service, sample_cv_content):
        cv_service.cv_content = sample_cv_content
        
        assert cv_service.has_cv() is True
    
    @pytest.mark.unit
    def test_has_cv_without_content(self, cv_service):
        cv_service.cv_content = None
        
        assert cv_service.has_cv() is False
    
    @pytest.mark.unit
    def test_has_cv_with_empty_content(self, cv_service):
        cv_service.cv_content = "   "
        
        assert cv_service.has_cv() is False
    
    @pytest.mark.unit
    def test_is_ready_when_initialized(self, cv_service):
        cv_service.is_initialized = True
        
        assert cv_service.is_ready() is True
    
    @pytest.mark.unit
    def test_is_ready_when_not_initialized(self, cv_service):
        cv_service.is_initialized = False
        
        assert cv_service.is_ready() is False
    
    @pytest.mark.unit
    def test_get_cv_info_with_cv(self, cv_service, sample_cv_content):
        cv_service.cv_content = sample_cv_content
        cv_service.last_loaded = datetime.now()
        
        with patch('pathlib.Path.exists', return_value=True):
            info = cv_service.get_cv_info()
        
        assert info["has_cv"] is True
        assert info["content_length"] == len(sample_cv_content)
        assert info["default_name"] == "Henrique Lobato"
        assert info["default_title"] == "Senior Python Developer"
        assert info["last_loaded"] is not None
        assert info["content_preview"] is not None
    
    @pytest.mark.unit
    def test_get_cv_info_without_cv(self, cv_service):
        cv_service.cv_content = None
        
        with patch('pathlib.Path.exists', return_value=False):
            info = cv_service.get_cv_info()
        
        assert info["has_cv"] is False
        assert info["content_length"] == 0
        assert info["content_preview"] is None
        assert info["last_loaded"] is None
        assert info["file_exists"] is False
    
    @pytest.mark.unit
    def test_get_cv_info_long_content_preview(self, cv_service):
        long_content = "A" * 500
        cv_service.cv_content = long_content
        
        with patch('pathlib.Path.exists', return_value=True):
            info = cv_service.get_cv_info()
        
        assert len(info["content_preview"]) <= 303
        assert info["content_preview"].endswith("...")
    
    @pytest.mark.unit
    def test_get_cv_content_with_content(self, cv_service, sample_cv_content):
        cv_service.cv_content = sample_cv_content
        
        content = cv_service.get_cv_content()
        
        assert content == sample_cv_content
    
    @pytest.mark.unit
    def test_get_cv_content_without_content(self, cv_service):
        cv_service.cv_content = None
        
        content = cv_service.get_cv_content()
        
        assert content is None
    
    @pytest.mark.unit
    def test_get_cv_stats_with_content(self, cv_service, sample_cv_content):
        cv_service.cv_content = sample_cv_content
        
        stats = cv_service.get_cv_stats()
        
        assert stats["character_count"] == len(sample_cv_content)
        assert stats["word_count"] > 0
        assert stats["line_count"] > 0
        assert stats["paragraph_count"] > 0
        assert isinstance(stats["character_count"], int)
        assert isinstance(stats["word_count"], int)
        assert isinstance(stats["line_count"], int)
        assert isinstance(stats["paragraph_count"], int)
    
    @pytest.mark.unit
    def test_get_cv_stats_without_content(self, cv_service):
        cv_service.cv_content = None
        
        stats = cv_service.get_cv_stats()
        
        assert stats["character_count"] == 0
        assert stats["word_count"] == 0
        assert stats["line_count"] == 0
        assert stats["paragraph_count"] == 0
    
    @pytest.mark.unit
    def test_get_cv_stats_empty_content(self, cv_service):
        cv_service.cv_content = ""
        
        stats = cv_service.get_cv_stats()
        
        assert stats["character_count"] == 0
        assert stats["word_count"] == 0
        assert stats["line_count"] == 0
        assert stats["paragraph_count"] == 0
    
    @pytest.mark.unit
    async def test_cleanup(self, cv_service, sample_cv_content):
        cv_service.cv_content = sample_cv_content
        cv_service.last_loaded = datetime.now()
        cv_service.is_initialized = True
        
        await cv_service.cleanup()
        
        assert cv_service.cv_content is None
        assert cv_service.last_loaded is None
        assert cv_service.is_initialized is False

    @pytest.mark.unit
    def test_get_system_prompt_with_cv_and_instructions(self, cv_service):
        """Test get_system_prompt with CV and custom instructions."""
        cv_service.cv_content = "John Doe\nSenior Developer\nExperienced in Python."
        cv_service.is_initialized = True
        
        custom_instructions = "Be professional and concise."
        result = cv_service.get_system_prompt(custom_instructions)
        
        assert "John Doe" in result
        assert "Senior Developer" in result
        assert custom_instructions in result

    @pytest.mark.unit
    def test_get_cv_info_with_cv_content(self, cv_service):
        """Test get_cv_info with CV content."""
        cv_service.cv_content = "Jane Smith\nData Scientist\nSpecializes in machine learning and data analysis."
        cv_service.is_initialized = True
        
        with patch('pathlib.Path.exists', return_value=True):
            result = cv_service.get_cv_info()
        
        assert result["has_cv"] is True
        assert result["default_name"] == "Jane Smith"
        assert result["default_title"] == "Data Scientist"
        assert "machine learning" in result["content_preview"]

    @pytest.mark.unit
    def test_get_cv_info_with_long_content(self, cv_service):
        """Test get_cv_info with long CV content that gets truncated."""
        # Create long content that will be truncated
        long_content = "John Doe\nSenior Developer\n" + "A" * 350 + " This is a very long CV content that should be truncated in the preview."
        cv_service.cv_content = long_content
        cv_service.is_initialized = True
        
        with patch('pathlib.Path.exists', return_value=True):
            result = cv_service.get_cv_info()
        
        assert result["has_cv"] is True
        assert len(result["content_preview"]) <= 303  # Should be truncated with "..."
        assert result["content_preview"].endswith("...")

    @pytest.mark.unit
    def test_get_cv_info_with_whitespace_only_content(self, cv_service):
        """Test get_cv_info with content that contains only whitespace (covers line 129->134)."""
        # Set content that has_cv() returns True for but results in empty lines after processing
        cv_service.cv_content = "\n\n   \n  \t  \n\n"  # Only whitespace and newlines
        cv_service.is_initialized = True
        
        with patch('pathlib.Path.exists', return_value=True):
            result = cv_service.get_cv_info()
        
        # has_cv() should return False for whitespace-only content
        # but if it somehow returns True, we want to test the lines processing
        assert result["has_cv"] is False  # This should be False due to has_cv() logic
        assert result["default_name"] == cv_service.default_name
        assert result["default_title"] == cv_service.default_title

    @pytest.mark.unit
    def test_get_cv_info_branch_coverage_empty_lines(self, cv_service):
        """Test get_cv_info to cover the branch where has_cv is True but lines is empty after processing."""
        # Force has_cv to return True by setting non-empty content
        cv_service.cv_content = "   "  # Non-empty but only spaces
        cv_service.is_initialized = True
        
        # Mock has_cv to return True to force the branch execution
        with patch.object(cv_service, 'has_cv', return_value=True), \
             patch('pathlib.Path.exists', return_value=True):
            
            result = cv_service.get_cv_info()
            
            # This should hit the branch where has_cv() and cv_content are True
            # but lines becomes empty after processing
            assert result["has_cv"] is True  # Due to our mock
            assert result["default_name"] == cv_service.default_name  # Should use defaults

    @pytest.mark.unit 
    def test_get_system_prompt_with_cv_available(self, cv_service):
        """Test get_system_prompt when CV is available (covers line 93->exit path)."""
        cv_service.cv_content = "Test User\nTest Title\nTest experience content."
        cv_service.is_initialized = True
        
        result = cv_service.get_system_prompt()
        
        # Should follow the else path (has_cv() returns True)
        assert "=== CV START ===" in result
        assert "=== CV END ===" in result
        assert cv_service.cv_content in result
        assert "Test User" in result


class TestCVServiceIntegration:
    """Integration tests for CV service with real file operations."""
    
    @pytest.mark.unit
    async def test_load_cv_with_temp_file(self, cv_service, temp_cv_file, sample_cv_content):
        """Test CV loading with a real temporary file."""
        cv_service.cv_file_path = temp_cv_file
        
        result = await cv_service.load_cv()
        
        assert result is True
        assert cv_service.cv_content == sample_cv_content
        assert cv_service.last_loaded is not None
    
    @pytest.mark.unit
    async def test_load_cv_with_empty_temp_file(self, cv_service, temp_empty_cv_file):
        """Test CV loading with an empty temporary file."""
        cv_service.cv_file_path = temp_empty_cv_file
        
        result = await cv_service.load_cv()
        
        assert result is False
        assert cv_service.cv_content is None
    
    @pytest.mark.unit
    async def test_full_workflow(self, cv_service, temp_cv_file, sample_cv_content):
        """Test complete CV service workflow."""
        cv_service.cv_file_path = temp_cv_file
        
        # Initialize
        await cv_service.initialize()
        
        # Check status
        assert cv_service.is_ready() is True
        assert cv_service.has_cv() is True
        
        # Get info
        info = cv_service.get_cv_info()
        assert info["has_cv"] is True
        assert info["content_length"] > 0
        
        # Get system prompt
        prompt = cv_service.get_system_prompt()
        assert "Henrique Lobato" in prompt or "HENRIQUE LOBATO" in prompt
        
        # Get stats
        stats = cv_service.get_cv_stats()
        assert stats["character_count"] > 0
        
        # Reload
        reload_result = await cv_service.reload_cv()
        assert reload_result is True
        
        # Cleanup
        await cv_service.cleanup()
        assert cv_service.is_ready() is False 