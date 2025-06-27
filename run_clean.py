#!/usr/bin/env python3
"""
Clean runner for Accent Recognition API
Suppresses verbose output and shows only essential information
"""

import subprocess
import sys
import os
import warnings
from pathlib import Path

def setup_clean_environment():
    """Setup environment to suppress verbose output"""
    # Suppress Python warnings
    warnings.filterwarnings("ignore")

    # Set environment variables to reduce verbosity
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["SPEECHBRAIN_CACHE"] = "pretrained_models"
    os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

    # Additional environment variables to suppress output
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

def check_virtual_environment():
    """Check if virtual environment is activated"""
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return True

    # Check if venv directory exists
    venv_path = Path("venv")
    if not venv_path.exists():
        return False

    return True

def run_clean_server():
    """Run uvicorn with clean output"""
    print("Accent Recognition API")
    print("=" * 40)

    # Check virtual environment
    if not check_virtual_environment():
        print("❌ Virtual environment not found or not activated!")
        print("\nPlease activate virtual environment first:")
        print("   Windows: venv\\Scripts\\activate")
        print("   Linux/macOS: source venv/bin/activate")
        print("\nThen run this script again.")
        return False

    print("✅ Virtual environment detected")
    print("Starting server...")

    # Determine python path based on OS
    if os.name == 'nt':  # Windows
        python_path = Path("venv/Scripts/python.exe")
        if not python_path.exists():
            python_path = "python"
    else:  # Linux/macOS
        python_path = Path("venv/bin/python")
        if not python_path.exists():
            python_path = "python"

    try:
        # Setup clean environment
        setup_clean_environment()

        print("API will be available at: http://localhost:8000")
        print("API documentation at: http://localhost:8000/docs")
        print("Health check at: http://localhost:8000/health")
        print("\nPress Ctrl+C to stop the server\n")
        
        # Start the server with minimal logging
        subprocess.run([
            str(python_path), "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "warning",  # Only show warnings and errors from uvicorn
            "--no-access-log"  # Disable access logs
        ], check=True)
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Server failed to start: {e}")
        return False
    except FileNotFoundError:
        print("Python or uvicorn not found. Make sure virtual environment is activated.")
        print("   Try: venv\\Scripts\\activate (Windows) or source venv/bin/activate (Linux/macOS)")
        return False
    
    return True

def main():
    """Main function"""
    if not run_clean_server():
        sys.exit(1)

if __name__ == "__main__":
    main()
