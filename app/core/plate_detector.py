import cv2
import numpy as np
from app.config import Config
import logging
import os

class PlateDetector:
    def __init__(self):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        self.cascade_path = Config.HAAR_CASCADE_PATH
        self.use_cascade = False
        self.plate_cascade = None
        self.load_detector()
    
    def load_detector(self):
        """Load Haar Cascade classifier"""
        try:
            # Check if cascade file exists
            if not os.path.exists(str(self.cascade_path)):
                self.logger.warning(f"Haar cascade file not found at {self.cascade_path}")
                self.logger.warning("Using contour detection method instead")
                self.use_cascade = False
                return
            
            self.plate_cascade = cv2.CascadeClassifier(str(self.cascade_path))
            if self.plate_cascade.empty():
                self.logger.warning("Haar cascade failed to load, using contour detection")
                self.use_cascade = False
            else:
                self.logger.info("Haar cascade loaded successfully")
                self.use_cascade = True
                
        except Exception as e:
            self.logger.error(f"Error loading cascade: {e}")
            self.logger.warning("Falling back to contour detection")
            self.use_cascade = False
    
    def preprocess_image(self, image):
        """Preprocess image for better plate detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for noise reduction
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(filtered)
        
        return equalized, gray
    
    def detect_with_cascade(self, gray_image):
        """Detect plates using Haar Cascade"""
        if not self.use_cascade or self.plate_cascade is None:
            return []
            
        plates = self.plate_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 30),
            maxSize=(400, 150)
        )
        return plates
    
    def detect_with_contours(self, image):
        """Detect plates using contour detection"""
        processed, gray = self.preprocess_image(image)
        
        # Edge detection
        edges = cv2.Canny(processed, 50, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        plates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if Config.MIN_PLATE_AREA < area < Config.MAX_PLATE_AREA:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (typical plate ratio is 2:1 to 5:1)
                aspect_ratio = w / h
                if 2.0 <= aspect_ratio <= 5.0:
                    plates.append((x, y, w, h))
        
        return plates
    
    def detect_plates(self, image):
        """Main plate detection method"""
        if self.use_cascade and self.plate_cascade is not None:
            processed, gray = self.preprocess_image(image)
            plates = self.detect_with_cascade(gray)
        else:
            plates = self.detect_with_contours(image)
        
        # Convert to list if it's a numpy array
        if isinstance(plates, np.ndarray):
            plates = plates.tolist()
        
        return plates
    
    def extract_plate_roi(self, image, plate_coords):
        """Extract plate region of interest"""
        x, y, w, h = plate_coords
        
        # Add some padding
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        roi = image[y_start:y_end, x_start:x_end]
        return roi
