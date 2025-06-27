#!/usr/bin/env python3
"""
Production deployment script for Accent Recognition API
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_virtual_environment():
    """Check if virtual environment exists"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found. Creating...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            return False
    else:
        print("✅ Virtual environment found")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Determine pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = Path("venv/Scripts/pip.exe")
    else:  # Linux/macOS
        pip_path = Path("venv/bin/pip")
    
    try:
        # Upgrade pip first
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_models():
    """Check if required models exist"""
    model_paths = [
        Path("pretrained_models"),
        Path("wav2vec2_checkpoints")
    ]
    
    for model_path in model_paths:
        if model_path.exists() and any(model_path.iterdir()):
            print(f"✅ Models found in {model_path}")
        else:
            print(f"⚠️ Models not found in {model_path} - will be downloaded on first run")
    
    return True

def run_health_check():
    """Run a quick health check"""
    print("🏥 Running health check...")
    
    # Determine python path based on OS
    if os.name == 'nt':  # Windows
        python_path = Path("venv/Scripts/python.exe")
    else:  # Linux/macOS
        python_path = Path("venv/bin/python")
    
    try:
        # Quick import test
        result = subprocess.run([
            str(python_path), "-c", 
            "import torch, torchaudio, speechbrain, fastapi; print('✅ All imports successful')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Health check timed out")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def start_production_server():
    """Start the production server"""
    print("🚀 Starting production server...")
    
    # Determine python path based on OS
    if os.name == 'nt':  # Windows
        python_path = Path("venv/Scripts/python.exe")
    else:  # Linux/macOS
        python_path = Path("venv/bin/python")
    
    try:
        print("📡 API will be available at: http://localhost:8000")
        print("📊 API documentation at: http://localhost:8000/docs")
        print("🔍 Health check at: http://localhost:8000/health")
        print("\n🛑 Press Ctrl+C to stop the server\n")
        
        # Start the server using uvicorn
        subprocess.run([
            str(python_path), "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return False
    
    return True

def main():
    """Main deployment function"""
    print("🎵 Accent Recognition API - Production Deployment")
    print("=" * 50)
    
    # Check requirements
    if not check_python_version():
        return False
    
    if not check_virtual_environment():
        return False
    
    if not install_dependencies():
        return False
    
    if not check_models():
        return False
    
    if not run_health_check():
        print("⚠️ Health check failed, but continuing...")
    
    print("\n✅ Deployment checks completed!")
    print("🚀 Ready to start production server")
    
    # Ask user if they want to start the server
    try:
        start_now = input("\nStart server now? (y/n): ").lower().strip()
        if start_now in ['y', 'yes']:
            return start_production_server()
        else:
            print("\n📝 To start the server later, run:")
            print("   uvicorn main:app --host 0.0.0.0 --port 8000")
            print("\n📚 For more information, see README.md")
            return True
    except KeyboardInterrupt:
        print("\n🛑 Deployment cancelled by user")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
