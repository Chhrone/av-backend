# 🚀 Cara Menjalankan Server

## Langkah-langkah Menjalankan Server

### 1. Aktifkan Virtual Environment
**PENTING: Selalu aktifkan virtual environment terlebih dahulu!**

```bash
# Windows
venv\Scripts\activate

# Linux/macOS  
source venv/bin/activate
```

Anda akan melihat `(venv)` di awal command prompt jika virtual environment sudah aktif.

### 2. Pilih Cara Menjalankan Server

#### Opsi A: Menggunakan `run_clean.py` (DIREKOMENDASIKAN)
**Output bersih tanpa log verbose**

```bash
python run_clean.py
```

**Keuntungan:**
- ✅ Output sangat bersih
- ✅ Hanya menampilkan informasi penting
- ✅ Menyembunyikan warning dan log verbose
- ✅ Otomatis cek virtual environment

#### Opsi B: Menggunakan uvicorn langsung
**Output lengkap dengan semua log**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Karakteristik:**
- ⚠️ Banyak log verbose
- ⚠️ Warning dari torchaudio, speechbrain, dll
- ✅ Menampilkan semua detail proses

## Output yang Diharapkan

### Dengan `run_clean.py`:
```
Accent Recognition API
========================================
✅ Virtual environment detected
Starting server...
API will be available at: http://localhost:8000
API documentation at: http://localhost:8000/docs
Health check at: http://localhost:8000/health
Press Ctrl+C to stop the server

The torchaudio backend is switched to 'soundfile'. Note that 'sox_io' is not supported on Windows.
The torchaudio backend is switched to 'soundfile'. Note that 'sox_io' is not supported on Windows.
Loading accent recognition model...
Model loaded and optimized in 3.23 seconds
Queue processor started
Cleanup system started
Initial cleanup completed
```

### Dengan uvicorn langsung:
```
The torchaudio backend is switched to 'soundfile'...
D:\...\speechbrain\utils\torch_audio_backend.py:22: UserWarning...
INFO:main:Loading model on device: cpu
INFO:speechbrain.pretrained.fetching:Fetch hyperparams.yaml...
[banyak log lainnya...]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Akses API

Setelah server berjalan, API dapat diakses di:
- **API Endpoint**: http://localhost:8000
- **Dokumentasi**: http://localhost:8000/docs  
- **Health Check**: http://localhost:8000/health

## Menghentikan Server

Tekan `Ctrl+C` untuk menghentikan server.

## Troubleshooting

### Error: Virtual environment not found
```bash
# Buat virtual environment baru
python -m venv venv

# Aktifkan
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Error: Python or uvicorn not found
Pastikan virtual environment sudah diaktifkan dan dependencies sudah terinstall.

### Port sudah digunakan
Jika port 8000 sudah digunakan, ubah port:
```bash
# Untuk run_clean.py, edit file dan ubah port
# Untuk uvicorn langsung:
uvicorn main:app --host 0.0.0.0 --port 8001
```
