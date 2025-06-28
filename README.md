# 🎵 Accent Recognition API

API untuk mendeteksi aksen Amerika (US) dari file audio menggunakan teknologi SpeechBrain dan wav2vec2.

## 🚀 Fitur Utama

- **In-Memory Processing** - File tidak disimpan ke disk
- **US Accent Detection** - Mengembalikan confidence score untuk aksen Amerika
- **High Performance** - Optimisasi dengan caching, queue system, dan GPU acceleration
- **Concurrent Requests** - Mendukung multiple requests bersamaan dengan sistem antrian
- **Auto Cleanup** - Sistem pembersihan file otomatis
- **Real-time Monitoring** - Statistik performance dan queue status

## 📋 Requirements

### System Requirements
- Python 3.8+
- Windows/Linux/macOS
- RAM: Minimum 4GB, Recommended 8GB+
- GPU: Optional (CUDA-compatible untuk performa lebih baik)

### Dependencies
```
fastapi>=0.104.1
uvicorn>=0.24.0
speechbrain>=0.5.16
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.35.0
python-multipart>=0.0.6
psutil>=5.9.0
```

## 🛠️ Installation & Deployment

### Quick Start (Automated)
```bash
# Clone repository
git clone <repository-url>
cd av-backend

# Run automated deployment
python deploy.py
```

### Manual Installation

#### 1. Clone Repository
```bash
git clone <repository-url>
cd av-backend
```

#### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

#### 3. Install Dependencies

**Option A: CPU Only (Default)**
```bash
pip install -r requirements.txt
```

**Option B: GPU/CUDA Support (Recommended untuk performa lebih baik)**

⚠️ **Penting**: Sesuaikan versi CUDA dengan GPU Anda!

```bash
# Untuk CUDA 11.8 (RTX 30xx, RTX 40xx series, dll)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Untuk CUDA 12.1 (GPU terbaru)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies lainnya
pip install -r requirements.txt
```

**Cara Cek Versi CUDA Anda:**
```bash
# Cek CUDA version
nvidia-smi

# Atau cek di Windows
nvcc --version
```

**Kompatibilitas GPU:**
- **CUDA 11.8**: RTX 20xx, RTX 30xx, RTX 40xx, GTX 16xx series
- **CUDA 12.1**: RTX 40xx series (terbaru), RTX 30xx, RTX 20xx
- **CPU Only**: Semua sistem (lebih lambat tapi tetap berfungsi)

💡 **Tips**: Jika tidak yakin, gunakan CUDA 11.8 karena lebih kompatibel dengan berbagai GPU.

#### 4. Download Models
Model akan otomatis didownload saat pertama kali menjalankan API.

## 🚀 Running the API

### Option 1: Automated Deployment (Recommended)
```bash
python deploy.py
```

### Option 2: Direct Uvicorn (Production)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Option 3: Production with Workers
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

API akan berjalan di: `http://localhost:8000`

## 📡 API Endpoints

### 1. **Accent Recognition** - `POST /identify`

**Request:**
```bash
curl -X POST "http://localhost:8000/identify" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav"
```

**Response:**
```json
{
  "us_confidence": 85.67
}
```

**Response Codes:**
- `200` - Success
- `400` - Invalid file format (only .wav allowed)
- `408` - Request timeout (30s)
- `503` - Server too busy (queue full)
- `500` - Internal server error

### 2. **Health Check** - `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "memory_usage_mb": 2048.5
}
```

### 3. **Performance Statistics** - `GET /stats`

**Response:**
```json
{
  "total_requests": 150,
  "average_inference_time": 1.234,
  "cache_hit_rate": 45.5,
  "memory_usage_mb": 2048.5,
  "cache_size": 25,
  "max_cache_size": 100,
  "queue_size": 3,
  "max_queue_size": 20,
  "active_requests": 5,
  "max_concurrent_requests": 5,
  "available_slots": 0,
  "cleanup_enabled": true,
  "cleanup_stats": {
    "last_cleanup": "2024-01-01T12:00:00",
    "total_cleanups": 15,
    "files_deleted": 0,
    "bytes_freed": 0
  }
}
```

### 4. **Queue Status** - `GET /queue-status`

**Response:**
```json
{
  "queue_size": 3,
  "max_queue_size": 20,
  "active_requests": 5,
  "max_concurrent_requests": 5,
  "available_slots": 0,
  "queue_full": false,
  "semaphore_locked": true,
  "active_request_ids": ["req_1_1703123456", "req_2_1703123457"]
}
```

### 5. **Cleanup Status** - `GET /cleanup-status`

**Response:**
```json
{
  "cleanup_enabled": true,
  "cleanup_interval": 300,
  "max_file_age_hours": 1,
  "uploads_folder": "uploads",
  "uploads_info": {
    "exists": true,
    "file_count": 0,
    "total_size": 0
  },
  "cleanup_stats": {
    "last_cleanup": "2024-01-01T12:00:00",
    "total_cleanups": 15,
    "files_deleted": 0,
    "bytes_freed": 0
  },
  "in_memory_processing": true,
  "note": "Files are processed in-memory, no disk storage used for uploads"
}
```

### 6. **Clear Cache** - `POST /clear-cache`

**Response:**
```json
{
  "message": "Cache cleared. Removed 25 entries.",
  "cache_info": {
    "hits": 150,
    "misses": 75,
    "maxsize": 100,
    "currsize": 0
  }
}
```

### 7. **Manual Cleanup** - `POST /cleanup-files`

**Response:**
```json
{
  "message": "Manual cleanup completed",
  "files_deleted": 0,
  "bytes_freed": 0,
  "cleanup_stats": {
    "last_cleanup": "2024-01-01T12:00:00",
    "total_cleanups": 16,
    "files_deleted": 0,
    "bytes_freed": 0
  }
}
```

### 8. **API Information** - `GET /`

**Response:**
```json
{
  "message": "Accent Recognition API",
  "version": "2.1.0",
  "response_format": {
    "identify_endpoint": {
      "description": "Returns only US accent confidence",
      "example": {"us_confidence": 85.67},
      "note": "Returns 0.0 if US accent not detected"
    }
  },
  "optimizations": [
    "In-memory file processing (NO disk storage)",
    "Result caching",
    "Model compilation (if supported)",
    "GPU acceleration (if available)",
    "Performance monitoring",
    "Request queuing system",
    "Concurrent request limiting",
    "Automatic file cleanup system"
  ],
  "queue_system": {
    "max_concurrent_requests": 5,
    "max_queue_size": 20,
    "request_timeout": 30
  },
  "endpoints": {
    "/identify": "POST - Get US accent confidence from audio file",
    "/health": "GET - Health check",
    "/stats": "GET - Performance statistics with queue and cleanup info",
    "/queue-status": "GET - Detailed queue status",
    "/cleanup-status": "GET - File cleanup system status",
    "/clear-cache": "POST - Clear cache",
    "/cleanup-files": "POST - Manually trigger file cleanup",
    "/": "GET - API information"
  }
}
```

## ⚙️ Configuration

Edit konfigurasi di `main.py`:

```python
class Config:
    # Model configuration
    MODEL_SOURCE = "Jzuluaga/accent-id-commonaccent_xlsr-en-english"
    CACHE_SIZE = 100  # Number of results to cache
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Queue configuration
    MAX_CONCURRENT_REQUESTS = 5  # Maximum concurrent inference requests
    MAX_QUEUE_SIZE = 20  # Maximum requests in queue
    REQUEST_TIMEOUT = 30  # Timeout per request in seconds
    
    # Cleanup configuration
    CLEANUP_INTERVAL = 300  # Cleanup every 5 minutes
    MAX_FILE_AGE_HOURS = 1  # Delete files older than 1 hour
    ENABLE_CLEANUP = True  # Enable/disable cleanup system
```

## 📁 File Structure

```
av-backend/
├── main.py                 # Main API application
├── requirements.txt        # Python dependencies
├── deploy.py              # Automated deployment script
├── README.md              # Documentation
├── .gitignore             # Git ignore rules
├── uploads/               # Upload folder (empty - in-memory processing)
│   └── .gitkeep          # Keep folder structure
├── pretrained_models/     # Downloaded models
├── wav2vec2_checkpoints/  # Model checkpoints
└── venv/                  # Virtual environment (created during setup)
```

## 🎯 Usage Examples

### Python Client
```python
import requests

# Upload audio file
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/identify',
        files={'file': f}
    )
    
result = response.json()
print(f"US Confidence: {result['us_confidence']}%")
```

### cURL
```bash
curl -X POST "http://localhost:8000/identify" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav"
```

### JavaScript/Fetch
```javascript
const formData = new FormData();
formData.append('file', audioFile);

fetch('http://localhost:8000/identify', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('US Confidence:', data.us_confidence + '%');
});
```

## 🔧 Performance Tuning

### For High Traffic
```python
# Increase concurrent requests
MAX_CONCURRENT_REQUESTS = 10

# Increase queue size
MAX_QUEUE_SIZE = 50

# Increase cache size
CACHE_SIZE = 200
```

### For Memory Optimization
```python
# Reduce cache size
CACHE_SIZE = 50

# Reduce concurrent requests
MAX_CONCURRENT_REQUESTS = 3
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Download Fails**
   ```bash
   # Check internet connection and try again
   python main.py
   ```

2. **CUDA Out of Memory**
   ```python
   # Reduce concurrent requests
   MAX_CONCURRENT_REQUESTS = 2
   ```

3. **High Memory Usage**
   ```bash
   # Clear cache periodically
   curl -X POST http://localhost:8000/clear-cache
   ```

4. **Queue Full Errors**
   ```python
   # Increase queue size or concurrent requests
   MAX_QUEUE_SIZE = 30
   MAX_CONCURRENT_REQUESTS = 8
   ```

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Performance Stats
```bash
curl http://localhost:8000/stats
```

### Queue Status
```bash
curl http://localhost:8000/queue-status
```

## 🔒 Security Notes

- File tidak disimpan ke disk (in-memory processing)
- Automatic cleanup untuk file residual
- File size limit (50MB default)
- Request timeout (30s default)
- Only .wav files accepted

## 📚 References and Citations

This project uses the following models and libraries. Please cite them appropriately:

### Accent Recognition Model
This project uses the accent classification model from:
- **Model**: [Jzuluaga/accent-id-commonaccent_xlsr-en-english](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english)
- **Paper**: CommonAccent: Exploring Large Acoustic Pretrained Models for Accent Classification Based on CommonVoice

```bibtex
@article{zuluaga2023commonaccent,
  title={CommonAccent: Exploring Large Acoustic Pretrained Models for Accent Classification Based on Common Voice},
  author={Zuluaga-Gomez, Juan and Ahmed, Sara and Visockas, Danielius and Subakan, Cem},
  journal={Interspeech 2023},
  url={https://arxiv.org/abs/2305.18283},
  year={2023}
}
```

### XLSR Model
The underlying XLSR model used for speech representation:

```bibtex
@article{conneau2020unsupervised,
  title={Unsupervised cross-lingual representation learning for speech recognition},
  author={Conneau, Alexis and Baevski, Alexei and Collobert, Ronan and Mohamed, Abdelrahman and Auli, Michael},
  journal={arXiv preprint arXiv:2006.13979},
  year={2020}
}
```

### SpeechBrain Framework
This project uses SpeechBrain for speech processing:

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

### Wav2Vec2 Model
The project uses Facebook's wav2vec2 model:

```bibtex
@article{baevski2020wav2vec,
  title={wav2vec 2.0: A framework for self-supervised learning of speech representations},
  author={Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={12449--12460},
  year={2020}
}
```

### Additional Libraries
- **FastAPI**: Modern web framework for building APIs
- **PyTorch**: Deep learning framework
- **Torchaudio**: Audio processing library for PyTorch
- **Uvicorn**: ASGI server implementation

## 🙏 Acknowledgments

- Thanks to Juan Pablo Zuluaga and team for the CommonAccent model
- Thanks to the SpeechBrain team for the excellent speech processing toolkit
- Thanks to Facebook AI Research for the wav2vec2 and XLSR models
- Thanks to the CommonVoice project for providing the training data

## 📝 License

This project is licensed under the MIT License.

**Note**: The pre-trained models used in this project have their own licenses. Please refer to their respective repositories for licensing information:
- [CommonAccent Model License](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english) - MIT License
- [SpeechBrain License](https://github.com/speechbrain/speechbrain/blob/develop/LICENSE) - Apache 2.0 License

## 🤝 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review the API documentation
