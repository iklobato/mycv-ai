"""
Comprehensive mocks for ML dependencies to enable testing of ML services.

This module provides mock implementations of:
- torch and related PyTorch components
- whisper and faster_whisper for speech recognition
- TTS for text-to-speech
- cv2 (OpenCV) for computer vision
- librosa for audio processing
- mediapipe for face detection
- soundfile for audio I/O
"""

import sys
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Any, Dict, List, Tuple, Optional, AsyncIterable


class MockTensor:
    """Mock PyTorch tensor."""
    def __init__(self, data=None, shape=(1,)):
        self.data = data or np.random.random(shape)
        self.shape = shape
        self.device = "cpu"
    
    def cuda(self):
        self.device = "cuda"
        return self
    
    def cpu(self):
        self.device = "cpu"
        return self
    
    def numpy(self):
        return self.data


class MockTorch:
    """Mock PyTorch module."""
    
    class MockCuda:
        @staticmethod
        def is_available():
            return False
    
    class MockBackends:
        class MockMPS:
            @staticmethod
            def is_available():
                return False
        
        mps = MockMPS()
    
    cuda = MockCuda()
    backends = MockBackends()
    
    @staticmethod
    def load(model_path, map_location=None):
        return {"model": "loaded"}
    
    @staticmethod
    def tensor(data):
        return MockTensor(data)


class MockWhisperModel:
    """Mock Whisper model."""
    
    def __init__(self, model_name="base", device="cpu"):
        self.model_name = model_name
        self.device = device
    
    def transcribe(self, audio, **kwargs):
        return {
            "text": "This is a mock transcription.",
            "language": "en",
            "segments": [
                {
                    "text": "This is a mock transcription.",
                    "start": 0.0,
                    "end": 2.5,
                    "confidence": 0.95
                }
            ]
        }


class MockSegment:
    """Mock segment for faster whisper."""
    def __init__(self, start, end, text, confidence=0.95):
        self.start = start
        self.end = end
        self.text = text
        self.confidence = confidence
        self.words = []


class MockFasterWhisperModel:
    """Mock Faster Whisper model."""
    
    def __init__(self, model_size_or_path="base", device="cpu", compute_type="int8", num_workers=1):
        self.model_size = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.num_workers = num_workers
    
    def transcribe(self, audio, **kwargs):
        segments = [
            MockSegment(0.0, 2.5, "This is a mock transcription.", 0.95)
        ]
        info = Mock()
        info.language = "en"
        info.language_probability = 0.98
        info.duration = 2.5
        return segments, info


class MockTTSModel:
    """Mock TTS model."""
    
    def __init__(self, model_name=None, progress_bar=False, gpu=False):
        self.model_name = model_name
        self.is_multi_speaker = True
        self.is_multi_lingual = True
        self.speakers = ["default", "speaker1", "speaker2"]
        self.languages = ["en", "es", "fr", "de"]
    
    def tts(self, text, speaker="default", language="en", **kwargs):
        # Return mock audio data (sine wave)
        duration = len(text) * 0.1  # Rough estimate
        sample_rate = 22050
        samples = int(duration * sample_rate)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        return audio
    
    def tts_to_file(self, text, file_path, speaker="default", language="en", **kwargs):
        # Mock writing audio to file
        audio = self.tts(text, speaker, language, **kwargs)
        # In real implementation, this would write to file
        return file_path


class MockMediaPipe:
    """Mock MediaPipe components."""
    
    class MockFaceMesh:
        def __init__(self, static_image_mode=False, max_num_faces=1, 
                     refine_landmarks=True, min_detection_confidence=0.5, 
                     min_tracking_confidence=0.5):
            pass
        
        def process(self, image):
            # Return mock landmarks
            mock_result = Mock()
            mock_result.multi_face_landmarks = [Mock()]
            return mock_result
    
    class MockFaceDetection:
        def __init__(self, model_selection=0, min_detection_confidence=0.5):
            pass
        
        def process(self, image):
            mock_result = Mock()
            mock_result.detections = [Mock()]
            return mock_result
    
    @property
    def solutions(self):
        solutions = Mock()
        solutions.face_mesh = Mock()
        solutions.face_mesh.FaceMesh = self.MockFaceMesh
        solutions.face_detection = Mock()
        solutions.face_detection.FaceDetection = self.MockFaceDetection
        return solutions


class MockCV2:
    """Mock OpenCV."""
    
    @staticmethod
    def imread(path, flags=None):
        # Return a mock image (random array)
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    @staticmethod
    def imwrite(path, img):
        return True
    
    @staticmethod
    def VideoCapture(source):
        mock_cap = Mock()
        mock_cap.read.return_value = (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        mock_cap.release.return_value = None
        mock_cap.isOpened.return_value = True
        return mock_cap
    
    @staticmethod
    def VideoWriter(filename, fourcc, fps, frameSize):
        mock_writer = Mock()
        mock_writer.write.return_value = None
        mock_writer.release.return_value = None
        return mock_writer
    
    @staticmethod
    def VideoWriter_fourcc(*args):
        return 1
    
    @staticmethod
    def circle(img, center, radius, color, thickness=-1):
        return img
    
    @staticmethod
    def ellipse(img, center, axes, angle, start_angle, end_angle, color, thickness=-1):
        return img
    
    @staticmethod
    def cvtColor(src, code):
        return src
    
    @staticmethod
    def imencode(ext, img):
        # Return success flag and mock encoded data
        return True, np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    
    @staticmethod
    def resize(img, size):
        # Return resized image
        height, width = size[1], size[0]
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


class MockLibrosa:
    """Mock librosa audio processing."""
    
    @staticmethod
    def load(path, sr=22050, mono=True):
        # Return mock audio array and sample rate
        duration = 3.0  # seconds
        samples = int(duration * sr)
        audio = np.random.random(samples).astype(np.float32)
        return audio, sr
    
    @staticmethod
    def resample(y, orig_sr, target_sr):
        # Mock resampling
        ratio = target_sr / orig_sr
        new_length = int(len(y) * ratio)
        return np.random.random(new_length).astype(np.float32)
    
    @property
    def util(self):
        util = Mock()
        util.normalize.return_value = np.random.random(1000).astype(np.float32)
        return util
    
    @property
    def effects(self):
        effects = Mock()
        audio = np.random.random(1000).astype(np.float32)
        effects.trim.return_value = (audio, np.array([0, len(audio)]))
        return effects


class MockSoundFile:
    """Mock soundfile."""
    
    @staticmethod
    def write(file, data, samplerate, **kwargs):
        return None
    
    @staticmethod
    def read(file, **kwargs):
        # Return mock audio data
        audio = np.random.random(22050).astype(np.float32)  # 1 second of audio
        return audio, 22050


class MockTorchAudio:
    """Mock torchaudio."""
    
    @staticmethod
    def load(path):
        # Return mock audio tensor and sample rate
        audio = MockTensor(np.random.random((1, 22050)), shape=(1, 22050))
        return audio, 22050
    
    @staticmethod
    def save(path, tensor, sample_rate):
        return None


class MockPydub:
    """Mock pydub AudioSegment."""
    
    class AudioSegment:
        def __init__(self, data=None):
            self.frame_rate = 22050
            self.channels = 1
            self.sample_width = 2
        
        @classmethod
        def from_file(cls, file, format=None):
            return cls()
        
        @classmethod
        def from_wav(cls, file):
            return cls()
        
        def export(self, out_f, format="wav", **kwargs):
            return Mock()
        
        def __len__(self):
            return 1000  # milliseconds
        
        def __add__(self, other):
            return self
        
        def __getitem__(self, key):
            return self


def setup_ml_mocks():
    """Set up all ML mocks in sys.modules."""
    
    # Add AsyncIterable to typing mock
    from typing import AsyncIterable
    
    # PyTorch mocks
    torch_mock = MockTorch()
    sys.modules['torch'] = torch_mock
    sys.modules['torchaudio'] = MockTorchAudio()
    sys.modules['torch.backends'] = torch_mock.backends
    sys.modules['torch.cuda'] = torch_mock.cuda
    
    # Whisper mocks
    whisper_mock = Mock()
    whisper_mock.load_model.return_value = MockWhisperModel()
    sys.modules['whisper'] = whisper_mock
    
    faster_whisper_mock = Mock()
    faster_whisper_mock.WhisperModel = MockFasterWhisperModel
    sys.modules['faster_whisper'] = faster_whisper_mock
    
    # TTS mocks
    tts_mock = Mock()
    tts_mock.api = Mock()
    tts_mock.api.TTS = MockTTSModel
    sys.modules['TTS'] = tts_mock
    sys.modules['TTS.api'] = tts_mock.api
    
    # OpenCV mock
    sys.modules['cv2'] = MockCV2()
    
    # Librosa mock
    sys.modules['librosa'] = MockLibrosa()
    
    # MediaPipe mock
    sys.modules['mediapipe'] = MockMediaPipe()
    
    # Audio processing mocks
    sys.modules['soundfile'] = MockSoundFile()
    
    # Pydub mock
    pydub_mock = Mock()
    pydub_mock.AudioSegment = MockPydub.AudioSegment
    sys.modules['pydub'] = pydub_mock
    
    # MoviePy mocks
    moviepy_mock = Mock()
    moviepy_mock.editor = Mock()
    moviepy_mock.editor.VideoFileClip = Mock
    moviepy_mock.editor.AudioFileClip = Mock
    sys.modules['moviepy'] = moviepy_mock
    sys.modules['moviepy.editor'] = moviepy_mock.editor
    
    # ImageIO mock
    imageio_mock = Mock()
    imageio_mock.mimsave.return_value = None
    imageio_mock.imread.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    sys.modules['imageio'] = imageio_mock


def cleanup_ml_mocks():
    """Remove ML mocks from sys.modules."""
    mock_modules = [
        'torch', 'torchaudio', 'torch.backends', 'torch.cuda',
        'whisper', 'faster_whisper', 'TTS', 'TTS.api',
        'cv2', 'librosa', 'mediapipe', 'soundfile', 'pydub',
        'moviepy', 'moviepy.editor', 'imageio'
    ]
    
    for module in mock_modules:
        if module in sys.modules:
            del sys.modules[module] 