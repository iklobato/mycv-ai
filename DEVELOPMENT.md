# Development Quick Reference

## Setup
```bash
uv sync --extra dev
./start_cv_avatar.sh
```

## Testing
```bash
# All tests
uv run pytest

# CV service only
uv run pytest tests/unit/test_cv_service.py

# With coverage
uv run pytest --cov=backend --cov-report=html
```

## Development Server
```bash
cd backend
uv run uvicorn main:app --reload --log-level debug
```

## Key Commands
```bash
# CV status
curl http://localhost:8000/cv

# Reload CV
curl -X POST http://localhost:8000/cv/reload

# Test response
curl -X POST http://localhost:8000/respond \
  -H "Content-Type: application/json" \
  -d '{"message": "What is your Python experience?"}'
```

## Code Quality
```bash
uv run black backend tests
uv run isort backend tests  
uv run flake8 backend
```

## Architecture
- `backend/services/cv_service.py` - CV management
- `backend/services/llm_service.py` - LLM + CV integration
- `backend/main.py` - FastAPI routes
- `tests/` - 122 tests with 90%+ coverage 