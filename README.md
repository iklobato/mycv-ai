# AI Avatar Video Call Application

A complete AI-powered web application that simulates a Google Meet-style interface where users can interact with an AI avatar in real-time using voice and video. **The AI avatar automatically uses Henrique Lobato's CV to respond as him in all conversations.**

## 🎯 Features

- **Real-time Voice Interaction**: Speak to your AI avatar and get responses
- **CV-Based Personality**: AI avatar automatically responds as Henrique Lobato based on his loaded CV
- **Google Meet-style Interface**: Familiar video call layout
- **AI Avatar Animation**: Lip-synced facial animation using SadTalker/Wav2Lip
- **Voice Cloning**: AI speaks with your cloned voice using XTTS-v2
- **Local AI Processing**: All AI models run locally (no cloud dependencies)
- **Webcam Support**: Optional user webcam preview
- **Real-time Audio Processing**: Low-latency speech-to-text and text-to-speech

## 🏗️ Architecture

```
├── frontend/           # Web interface (HTML/CSS/JS)
├── backend/           # FastAPI server
├── models/            # AI model downloads and scripts
├── avatar_photos/     # Your photos for avatar creation
├── voice_samples/     # Voice samples for TTS training
├── data/              # CV and configuration files
│   └── cv.txt         # Henrique's CV (auto-loaded)
├── docker/            # Docker configuration
└── requirements.txt   # Python dependencies
```

## 🛠️ Tech Stack

### Frontend
- HTML5, CSS3, JavaScript
- WebRTC for media access
- WebSocket for real-time communication

### Backend
- **FastAPI**: Web server and API
- **CV Service**: Automatic CV loading and personality injection
- **Whisper**: Speech-to-text transcription
- **Ollama**: Local LLM hosting (Llama3/Mistral/Phi-3) with CV context
- **XTTS-v2**: Voice cloning and text-to-speech
- **SadTalker**: Avatar animation and lip-sync

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- NVIDIA GPU (recommended for faster inference)
- 16GB+ RAM
- 50GB+ free disk space

### Installation

1. **Clone and Setup**
```bash
git clone <repository>
cd mycv-ai
```

2. **Install Dependencies**
```bash
pip install uv
uv pip install -r requirements.txt
```

3. **Setup CV (Important!)**
```bash
# The CV file is automatically created at data/cv.txt
# Edit this file with your actual CV content for the AI avatar personality
nano data/cv.txt
```

4. **Download AI Models**
```bash
python models/download_models.py
```

5. **Start Ollama and Pull LLM**
```bash
ollama serve
ollama pull llama3
```

6. **Run with Docker**
```bash
docker-compose up --build
```

7. **Access Application**
Open `http://localhost:8000` in your browser

## 🧠 CV-Based Personality System

### How It Works
1. **Automatic Loading**: On startup, the system reads `data/cv.txt`
2. **Personality Injection**: Every LLM request includes the CV as system context
3. **Consistent Responses**: AI avatar always responds as Henrique based on CV content
4. **Professional Context**: Perfect for HR interviews, networking, or showcasing skills

### CV File Format
The `data/cv.txt` file should contain:
- **Personal Information**: Name, title, contact preferences
- **Professional Experience**: Detailed work history
- **Technical Skills**: Programming languages, frameworks, tools
- **Projects**: Key accomplishments and technologies used
- **Education & Certifications**: Academic background
- **Personality Traits**: Communication style and interests

### Example System Prompt (Auto-Generated)
```
You are Henrique Lobato, a Senior Python Developer. You must respond as this person in all conversations.

When answering questions about experience, skills, projects, or any professional matters, base your responses strictly on the CV information below.

=== CV START ===
[Full CV content automatically inserted here]
=== CV END ===

Important guidelines:
- Always respond as Henrique Lobato
- Use the CV information as your knowledge base
- Speak naturally and conversationally
- Keep responses concise for voice conversation
```

## 📁 Project Structure

### API Endpoints

#### Core Endpoints
- `POST /transcribe`: Audio → Text transcription
- `POST /respond`: Text → LLM response (with CV context)
- `POST /speak`: Text → Audio with voice cloning
- `POST /animate`: Audio → Animated avatar video
- `WebSocket /ws`: Real-time communication

#### CV Management Endpoints
- `GET /cv`: Get CV information and status (debug)
- `POST /cv/reload`: Reload CV from file
- `GET /models/status`: Get all model status including CV integration

### Model Pipeline

1. **Audio Input** → Whisper → **Text Transcription**
2. **Text + CV Context** → Ollama LLM → **Personalized AI Response**
3. **AI Response** → XTTS-v2 → **Cloned Voice Audio**
4. **Audio + Photo** → SadTalker → **Animated Avatar**
5. **Avatar Video** → Frontend → **Live Playback**

## 🎭 Avatar Setup

1. **Edit CV**: Update `data/cv.txt` with your actual professional information
2. **Add Your Photos**: Place 5-10 front-facing photos in `avatar_photos/`
3. **Record Voice Samples**: Add 2-3 minutes of your voice in `voice_samples/`
4. **Train Voice Model**: Run `python backend/train_voice.py`
5. **Test Personality**: Use `/cv` endpoint to verify CV is loaded correctly

## ⚙️ Configuration

### Environment Variables
```bash
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=llama3
VOICE_MODEL_PATH=./models/xtts_v2
AVATAR_MODEL_PATH=./models/sadtalker
```

### CV Service Configuration
- **CV File Path**: `./data/cv.txt` (automatically created if missing)
- **Auto-reload**: CV is loaded once at startup
- **Fallback**: If CV is missing, uses generic personality
- **Validation**: Minimum 50 characters required for CV content

### Performance Tuning
- **GPU Memory**: Adjust batch sizes in `backend/config.py`
- **Audio Quality**: Modify sample rates in `backend/audio_config.py`
- **Response Time**: Configure chunk sizes for real-time processing

## 🐳 Docker Deployment

```bash
# Development
docker-compose -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.prod.yml up
```

## 🧪 Testing CV Integration

### Debug CV Status
```bash
curl http://localhost:8000/cv
```

### Test Personality Response
```bash
curl -X POST http://localhost:8000/respond \
  -H "Content-Type: application/json" \
  -d '{"text": "Tell me about your Python experience"}'
```

### Reload CV (if updated)
```bash
curl -X POST http://localhost:8000/cv/reload
```

### Check Model Status
```bash
curl http://localhost:8000/models/status
```

## 📝 Usage

1. **Open Application**: Navigate to `http://localhost:8000`
2. **Grant Permissions**: Allow microphone and camera access
3. **Start Conversation**: Click the microphone button and speak
4. **Professional Interaction**: The AI avatar responds as Henrique based on his CV
5. **Ask Technical Questions**: Test knowledge about Python, AI, projects
6. **HR Simulation**: Perfect for practicing interviews or showcasing skills

## 🔧 Troubleshooting

### CV-Related Issues
- **CV Not Loading**: Check `data/cv.txt` exists and has content
- **Generic Responses**: Verify CV service status with `/cv` endpoint
- **Update CV**: Edit `data/cv.txt` and call `/cv/reload`
- **Missing Personality**: Check logs for CV loading errors

### Common Issues
- **CUDA Out of Memory**: Reduce batch sizes or use CPU inference
- **Audio Latency**: Check microphone settings and buffer sizes
- **Avatar Animation Lag**: Ensure sufficient GPU memory
- **Model Download Failures**: Check internet connection and disk space

### Performance Optimization
- Use quantized models for faster inference
- Enable mixed precision training
- Optimize WebRTC settings for lower latency

## 🔍 Example Conversations

With the CV loaded, the AI avatar can respond to:

**Technical Questions:**
- "What programming languages do you know?"
- "Tell me about your AI projects"
- "What's your experience with Django?"

**Professional Questions:**
- "What's your current role?"
- "Describe your biggest accomplishment"
- "What technologies excite you most?"

**Project Details:**
- "Tell me about the AI Avatar project"
- "How did you implement real-time communication?"
- "What challenges did you face with voice cloning?"

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Update CV in `data/cv.txt` if needed
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Whisper**: OpenAI for speech recognition
- **Ollama**: Local LLM hosting
- **XTTS-v2**: Coqui TTS for voice cloning
- **SadTalker**: Avatar animation technology
- **FastAPI**: High-performance web framework 

## 🎯 Pro Tips

- **Update CV Regularly**: Keep `data/cv.txt` current for accurate responses
- **Test Responses**: Use the `/cv` endpoint to verify personality injection
- **Monitor Logs**: Check startup logs for CV loading status
- **Backup CV**: Keep a backup of your `data/cv.txt` file
- **Customize Prompts**: The CV context can be extended with additional instructions 