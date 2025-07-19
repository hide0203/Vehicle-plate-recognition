import os
from pathlib import Path

class Config:
    # Base directory
    BASE_DIR = Path(__file__).parent.parent
    
    # Database configuration
    DATABASE_PATH = BASE_DIR / "data" / "plates.db"
    
    # Camera settings
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30
    
    # OCR settings
    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows
    # TESSERACT_CMD = '/usr/bin/tesseract'  # Linux/Mac
    
    # Plate detection settings
    MIN_PLATE_AREA = 2000
    MAX_PLATE_AREA = 50000
    PLATE_CONFIDENCE_THRESHOLD = 0.5
    
    # Storage paths
    CAPTURED_PLATES_DIR = BASE_DIR / "data" / "captured_plates"
    MODELS_DIR = BASE_DIR / "data" / "models"
    HAAR_CASCADE_PATH = BASE_DIR / "data" / "haarcascades" / "haarcascade_russian_plate_number.xml"
    
    # Web settings
    SECRET_KEY = "your-secret-key-here"
    DEBUG = True
    HOST = "0.0.0.0"
    PORT = 5000
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.CAPTURED_PLATES_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        (cls.BASE_DIR / "logs").mkdir(exist_ok=True)
