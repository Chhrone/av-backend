from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import starlette.middleware.base as middleware_base
import os
import shutil
import torch
import torchaudio
from speechbrain.pretrained.interfaces import foreign_class
import asyncio
import time
import hashlib
import io
import os
import glob
import tempfile
from functools import lru_cache
from typing import Optional, Tuple
import logging
import psutil
import gc
from asyncio import Semaphore, Queue
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# Setup clean logging configuration
import warnings
import sys

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.set_audio_backend has been deprecated.*")
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")
warnings.filterwarnings("ignore", message=".*resume_download.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Some weights of Wav2Vec2Model were not initialized.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")

# Setup minimal logging - only show errors and critical info
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Suppress all verbose loggers
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torchaudio").setLevel(logging.ERROR)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)  # Disable access logs

logger = logging.getLogger(__name__)

# Inisialisasi FastAPI
app = FastAPI(title="Accent Recognition API")

# Request logging middleware
class RequestLoggingMiddleware(middleware_base.BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        import time

        # Log incoming request
        start_time = time.time()
        timestamp = time.strftime("%H:%M:%S")

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Log request
        if request.url.path == "/identify":
            print(f"[{timestamp}] REQUEST {request.method} {request.url.path} from {client_ip}")
        elif request.url.path == "/health":
            print(f"[{timestamp}] HEALTH {request.method} {request.url.path} from {client_ip}")
        elif request.url.path == "/docs" or request.url.path == "/openapi.json":
            print(f"[{timestamp}] DOCS {request.method} {request.url.path} from {client_ip}")
        elif request.url.path == "/":
            print(f"[{timestamp}] ROOT {request.method} {request.url.path} from {client_ip}")
        elif not request.url.path.startswith("/static"):  # Skip static files
            print(f"[{timestamp}] API {request.method} {request.url.path} from {client_ip}")

        # Process request
        response = await call_next(request)

        # Log response
        process_time = time.time() - start_time
        timestamp = time.strftime("%H:%M:%S")

        if request.url.path == "/identify":
            if response.status_code == 200:
                print(f"[{timestamp}] RESPONSE: {response.status_code} OK ({process_time:.3f}s)")
            else:
                print(f"[{timestamp}] RESPONSE: {response.status_code} ERROR ({process_time:.3f}s)")
        elif not request.url.path.startswith("/static") and request.url.path not in ["/docs", "/openapi.json"]:
            if response.status_code == 200:
                print(f"[{timestamp}] RESPONSE: {response.status_code} OK ({process_time:.3f}s)")
            else:
                print(f"[{timestamp}] RESPONSE: {response.status_code} ERROR ({process_time:.3f}s)")

        return response

# Add middleware
app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    MODEL_SOURCE = "Jzuluaga/accent-id-commonaccent_xlsr-en-english"
    PYMODULE_FILE = "custom_interface.py"
    CLASSNAME = "CustomEncoderWav2vec2Classifier"
    CACHE_SIZE = 100  # Number of results to cache
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Queue configuration
    MAX_CONCURRENT_REQUESTS = 5  # Maximum concurrent inference requests
    MAX_QUEUE_SIZE = 20  # Maximum requests in queue
    REQUEST_TIMEOUT = 30  # Timeout per request in seconds

    # Cleanup configuration
    CLEANUP_INTERVAL = 300  # Cleanup every 5 minutes (300 seconds)
    UPLOADS_FOLDER = "uploads"  # Folder to clean
    MAX_FILE_AGE_HOURS = 1  # Delete files older than 1 hour
    ENABLE_CLEANUP = True  # Enable/disable cleanup system

config = Config()

# Global variables for model, cache, and queue
classifier = None
result_cache = {}

# Request queue system
@dataclass
class QueuedRequest:
    request_id: str
    file_content: bytes
    timestamp: datetime
    future: asyncio.Future

# Global queue and semaphore for request limiting
request_semaphore = Semaphore(config.MAX_CONCURRENT_REQUESTS)
request_queue = Queue(maxsize=config.MAX_QUEUE_SIZE)
active_requests = {}
request_counter = 0

# Cleanup statistics
cleanup_stats = {
    "last_cleanup": None,
    "total_cleanups": 0,
    "files_deleted": 0,
    "bytes_freed": 0
}

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.request_count = 0
        self.total_inference_time = 0.0
        self.cache_hits = 0

    def log_request(self, inference_time: float, cache_hit: bool = False):
        self.request_count += 1
        self.total_inference_time += inference_time
        if cache_hit:
            self.cache_hits += 1

    def get_stats(self):
        avg_time = self.total_inference_time / max(self.request_count, 1)
        cache_hit_rate = self.cache_hits / max(self.request_count, 1) * 100
        return {
            "total_requests": self.request_count,
            "average_inference_time": round(avg_time, 3),
            "cache_hit_rate": round(cache_hit_rate, 2),
            "memory_usage_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
        }

monitor = PerformanceMonitor()

def load_and_optimize_model():
    """Load model with optimizations"""
    global classifier

    # Suppress verbose output during model loading
    print("Loading accent recognition model...")
    start_time = time.time()

    try:
        # Temporarily redirect stdout to suppress verbose model loading
        import contextlib
        from io import StringIO

        with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
            classifier = foreign_class(
                source=config.MODEL_SOURCE,
                pymodule_file=config.PYMODULE_FILE,
                classname=config.CLASSNAME
            )

        # Move model to appropriate device
        if hasattr(classifier, 'to'):
            classifier = classifier.to(config.DEVICE)

        # Enable optimizations for PyTorch 2.0+
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
            try:
                # Compile the model for faster inference
                if hasattr(classifier, 'mods'):
                    if hasattr(classifier.mods, 'wav2vec2'):
                        classifier.mods.wav2vec2 = torch.compile(classifier.mods.wav2vec2)
                    if hasattr(classifier.mods, 'output_mlp'):
                        classifier.mods.output_mlp = torch.compile(classifier.mods.output_mlp)
            except Exception:
                pass  # Silently ignore compilation failures

        # Set model to evaluation mode
        classifier.eval()

        # Warm up the model with a dummy input
        try:
            dummy_audio = torch.randn(1, 16000).to(config.DEVICE)
            with torch.no_grad():
                _ = classifier.encode_batch(dummy_audio)
        except Exception:
            pass  # Silently ignore warmup failures

        load_time = time.time() - start_time
        print(f"Model loaded and optimized in {load_time:.2f} seconds")

    except Exception as e:
        print(f"Model loading failed: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

# Load model saat aplikasi start-up
load_and_optimize_model()

# Global variables for background tasks
queue_processor_task = None
cleanup_task = None

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    global queue_processor_task, cleanup_task

    # Start queue processor
    queue_processor_task = asyncio.create_task(process_request_queue())
    print("Queue processor started")

    # Start cleanup task if enabled
    if config.ENABLE_CLEANUP:
        cleanup_task = asyncio.create_task(periodic_cleanup())
        print("Cleanup system started")

        # Run initial cleanup
        cleanup_old_files()
        print("Initial cleanup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    global queue_processor_task, cleanup_task

    # Stop queue processor
    if queue_processor_task:
        queue_processor_task.cancel()
        try:
            await queue_processor_task
        except asyncio.CancelledError:
            pass
        pass  # Silently stop queue processor

    # Stop cleanup task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

def calculate_file_hash(file_content: bytes) -> str:
    """Calculate MD5 hash of file content for caching"""
    return hashlib.md5(file_content).hexdigest()

def cleanup_old_files():
    """Clean up old files from uploads folder and temp directories"""
    if not config.ENABLE_CLEANUP:
        return

    try:
        files_deleted = 0
        bytes_freed = 0
        cutoff_time = datetime.now() - timedelta(hours=config.MAX_FILE_AGE_HOURS)

        # Clean uploads folder
        uploads_path = Path(config.UPLOADS_FOLDER)
        if uploads_path.exists():
            for file_path in uploads_path.glob("*"):
                if file_path.is_file():
                    file_stat = file_path.stat()
                    file_modified = datetime.fromtimestamp(file_stat.st_mtime)

                    if file_modified < cutoff_time:
                        file_size = file_stat.st_size
                        try:
                            file_path.unlink()
                            files_deleted += 1
                            bytes_freed += file_size
                            pass  # Silently delete files
                        except Exception:
                            pass  # Silently ignore deletion errors

        # Clean system temp files related to our app (optional)
        temp_dir = Path(tempfile.gettempdir())
        for temp_file in temp_dir.glob("tmp*"):
            if temp_file.is_file():
                try:
                    file_stat = temp_file.stat()
                    file_modified = datetime.fromtimestamp(file_stat.st_mtime)

                    if file_modified < cutoff_time:
                        file_size = file_stat.st_size
                        temp_file.unlink()
                        files_deleted += 1
                        bytes_freed += file_size
                except Exception:
                    # Ignore temp file cleanup errors
                    pass

        # Update cleanup stats
        cleanup_stats["last_cleanup"] = datetime.now()
        cleanup_stats["total_cleanups"] += 1
        cleanup_stats["files_deleted"] += files_deleted
        cleanup_stats["bytes_freed"] += bytes_freed

        # Silently complete cleanup

    except Exception:
        pass  # Silently ignore cleanup errors

async def periodic_cleanup():
    """Background task for periodic cleanup"""
    while True:
        try:
            await asyncio.sleep(config.CLEANUP_INTERVAL)
            cleanup_old_files()
        except Exception:
            await asyncio.sleep(60)  # Wait 1 minute before retrying

@lru_cache(maxsize=config.CACHE_SIZE)
def cached_inference(file_hash: str, file_content_bytes: bytes) -> float:
    """Cached inference function - returns only US confidence"""
    # Convert bytes back to audio tensor
    audio_buffer = io.BytesIO(file_content_bytes)

    # Load audio directly from memory
    waveform, sample_rate = torchaudio.load(audio_buffer, format="wav")

    # Ensure correct sample rate (16kHz for wav2vec2)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Move to device
    waveform = waveform.to(config.DEVICE)

    # Perform inference
    with torch.no_grad():
        # Use encode_batch and output_mlp directly for better performance
        outputs = classifier.encode_batch(waveform)
        outputs = classifier.mods.output_mlp(outputs).squeeze(1)
        out_prob = classifier.hparams.softmax(outputs)

        # Extract US confidence directly
        us_confidence = get_us_confidence_from_prob(out_prob)

    return us_confidence

def get_us_confidence_from_prob(out_prob: torch.Tensor) -> float:
    """Extract US confidence score from probabilities"""
    try:
        labels = classifier.label_encoder.decode_ndim(torch.arange(len(out_prob)))
        us_index = None
        for i, label in enumerate(labels):
            if label.lower() == "us":
                us_index = i
                break

        if us_index is not None:
            us_confidence = float(out_prob[us_index])
            return round(us_confidence * 100, 2)  # Convert to percentage
        return 0.0  # Return 0 if US not found
    except Exception:
        return 0.0

async def process_request_queue():
    """Background task to process queued requests"""
    while True:
        try:
            # Get request from queue
            queued_request = await request_queue.get()

            # Check if request is still valid (not timed out)
            if queued_request.future.cancelled():
                request_queue.task_done()
                continue

            # Acquire semaphore for concurrent request limiting
            async with request_semaphore:
                try:
                    # Process the request
                    file_hash = calculate_file_hash(queued_request.file_content)

                    # Check cache first
                    if file_hash in result_cache:
                        result = result_cache[file_hash]
                    else:
                        # Perform inference
                        us_confidence = cached_inference(file_hash, queued_request.file_content)
                        result = {"us_confidence": us_confidence}

                        # Cache the result
                        if len(result_cache) >= config.CACHE_SIZE:
                            # Remove oldest entry (simple FIFO)
                            oldest_key = next(iter(result_cache))
                            del result_cache[oldest_key]

                        result_cache[file_hash] = result

                    # Set the result
                    if not queued_request.future.cancelled():
                        queued_request.future.set_result(result)

                except Exception as e:
                    if not queued_request.future.cancelled():
                        queued_request.future.set_exception(e)

                finally:
                    # Remove from active requests
                    active_requests.pop(queued_request.request_id, None)
                    request_queue.task_done()

        except Exception:
            await asyncio.sleep(1)

# Queue processor will be started in startup event

@app.post("/identify")
async def identify_accent(file: UploadFile = File(...)):
    global request_counter
    start_time = time.time()

    try:
        # Validasi jenis file
        if not file.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="Only .wav files are allowed")

        # Validasi ukuran file
        file_content = await file.read()
        if len(file_content) > config.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")

        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Calculate file hash for caching
        file_hash = calculate_file_hash(file_content)

        # Check cache first
        cache_hit = file_hash in result_cache
        if cache_hit:
            result = result_cache[file_hash]
            inference_time = time.time() - start_time
            monitor.log_request(inference_time, cache_hit=True)
            return JSONResponse(content=result)

        # Check if we can process immediately or need to queue
        if request_semaphore.locked() or request_queue.qsize() > 0:
            # Queue is needed
            if request_queue.full():
                raise HTTPException(status_code=503, detail="Server too busy. Please try again later.")

            # Create queued request
            request_counter += 1
            request_id = f"req_{request_counter}_{int(time.time())}"
            future = asyncio.Future()

            queued_request = QueuedRequest(
                request_id=request_id,
                file_content=file_content,
                timestamp=datetime.now(),
                future=future
            )

            # Add to queue and active requests
            await request_queue.put(queued_request)
            active_requests[request_id] = queued_request

            # Wait for result with timeout
            try:
                result = await asyncio.wait_for(future, timeout=config.REQUEST_TIMEOUT)
                inference_time = time.time() - start_time
                monitor.log_request(inference_time, cache_hit=False)
                return JSONResponse(content=result)

            except asyncio.TimeoutError:
                # Cancel the request
                future.cancel()
                active_requests.pop(request_id, None)
                raise HTTPException(status_code=408, detail="Request timeout")

        else:
            # Process immediately
            async with request_semaphore:
                try:
                    us_confidence = cached_inference(file_hash, file_content)
                    result = {"us_confidence": us_confidence}

                    # Cache the result
                    if len(result_cache) >= config.CACHE_SIZE:
                        # Remove oldest entry (simple FIFO)
                        oldest_key = next(iter(result_cache))
                        del result_cache[oldest_key]

                    result_cache[file_hash] = result

                    inference_time = time.time() - start_time
                    monitor.log_request(inference_time, cache_hit=False)
                    return JSONResponse(content=result)

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        # Force garbage collection to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "device": config.DEVICE,
        "torch_version": torch.__version__
    }

@app.get("/stats")
async def get_performance_stats():
    """Get performance statistics including queue and cleanup info"""
    stats = monitor.get_stats()
    stats.update({
        "cache_size": len(result_cache),
        "max_cache_size": config.CACHE_SIZE,
        "device": config.DEVICE,
        "queue_size": request_queue.qsize(),
        "max_queue_size": config.MAX_QUEUE_SIZE,
        "active_requests": len(active_requests),
        "max_concurrent_requests": config.MAX_CONCURRENT_REQUESTS,
        "available_slots": config.MAX_CONCURRENT_REQUESTS - len(active_requests),
        "cleanup_enabled": config.ENABLE_CLEANUP,
        "cleanup_stats": cleanup_stats.copy()
    })
    return stats

@app.get("/queue-status")
async def get_queue_status():
    """Get detailed queue status"""
    return {
        "queue_size": request_queue.qsize(),
        "max_queue_size": config.MAX_QUEUE_SIZE,
        "active_requests": len(active_requests),
        "max_concurrent_requests": config.MAX_CONCURRENT_REQUESTS,
        "available_slots": config.MAX_CONCURRENT_REQUESTS - len(active_requests),
        "queue_full": request_queue.full(),
        "semaphore_locked": request_semaphore.locked(),
        "active_request_ids": list(active_requests.keys())
    }

@app.post("/clear-cache")
async def clear_cache():
    """Clear the result cache"""
    global result_cache
    cache_size = len(result_cache)
    result_cache.clear()

    # Clear LRU cache
    cached_inference.cache_clear()

    # Force garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "message": f"Cache cleared. Removed {cache_size} entries.",
        "cache_info": cached_inference.cache_info()._asdict()
    }

@app.post("/cleanup-files")
async def manual_cleanup():
    """Manually trigger file cleanup"""
    if not config.ENABLE_CLEANUP:
        return {
            "message": "Cleanup system is disabled",
            "cleanup_enabled": False
        }

    files_before = cleanup_stats["files_deleted"]
    bytes_before = cleanup_stats["bytes_freed"]

    cleanup_old_files()

    files_cleaned = cleanup_stats["files_deleted"] - files_before
    bytes_cleaned = cleanup_stats["bytes_freed"] - bytes_before

    return {
        "message": "Manual cleanup completed",
        "files_deleted": files_cleaned,
        "bytes_freed": bytes_cleaned,
        "cleanup_stats": cleanup_stats.copy()
    }

@app.get("/cleanup-status")
async def get_cleanup_status():
    """Get cleanup system status"""
    uploads_path = Path(config.UPLOADS_FOLDER)
    uploads_info = {
        "exists": uploads_path.exists(),
        "file_count": 0,
        "total_size": 0
    }

    if uploads_path.exists():
        files = list(uploads_path.glob("*"))
        uploads_info["file_count"] = len([f for f in files if f.is_file()])
        uploads_info["total_size"] = sum(f.stat().st_size for f in files if f.is_file())

    return {
        "cleanup_enabled": config.ENABLE_CLEANUP,
        "cleanup_interval": config.CLEANUP_INTERVAL,
        "max_file_age_hours": config.MAX_FILE_AGE_HOURS,
        "uploads_folder": config.UPLOADS_FOLDER,
        "uploads_info": uploads_info,
        "cleanup_stats": cleanup_stats.copy(),
        "in_memory_processing": True,
        "note": "Files are processed in-memory, no disk storage used for uploads"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
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
            "max_concurrent_requests": config.MAX_CONCURRENT_REQUESTS,
            "max_queue_size": config.MAX_QUEUE_SIZE,
            "request_timeout": config.REQUEST_TIMEOUT
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