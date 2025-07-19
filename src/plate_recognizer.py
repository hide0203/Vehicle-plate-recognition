"""
License Plate Text Recognition Module
Uses EasyOCR for text extraction and recognition
"""

import cv2
import numpy as np
import easyocr
import re
from typing import Optional

class PlateRecognizer:
    def __init__(self, config):
        self.config = config
        self.reader = easyocr.Reader(['en'], gpu=config.get('use_gpu', False))
        self.confidence_threshold = config['recognition']['confidence_threshold']
        self.last_confidence = 0.0
        
    def recognize_text(self, plate_image: np.ndarray) -> str:
        """
        Extract text from license plate image
        Returns cleaned plate number string
        """
        # Preprocess image for better OCR
        processed_image = self._preprocess_for_ocr(plate_image)
        
        # Perform OCR
        results = self.reader.readtext(processed_image)
        
        # Process results
        best_text = ""
        best_confidence = 0.0
        
        for (bbox, text, confidence) in results:
            if confidence > best_confidence and confidence > self.confidence_threshold:
                best_text = text
                best_confidence = confidence
        
        self.last_confidence = best_confidence
        
        # Clean and validate the text
        cleaned_text = self._clean_plate_text(best_text)
        
        return cleaned_text
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale_factor = 50 / height
            new_width = int(width * scale_factor)
            gray = cv2.resize(gray, (new_width, 50), interpolation=cv2.INTER_CUBIC)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Invert if needed (make text black on white background)
        if np.mean(thresh) < 127:
            thresh = cv2.bitwise_not(thresh)
        
        return thresh
    
    def _clean_plate_text(self, text: str) -> str:
        """Clean and validate plate number text"""
        if not text:
            return ""
        
        # Remove special characters and spaces
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Common OCR corrections
        corrections = {
            'O': '0', 'I': '1', 'S': '5', 'Z': '2',
            'B': '8', 'G': '6', 'Q': '0'
        }
        
        # Apply corrections based on position (letters usually at start, numbers at end)
        corrected = ""
        for i, char in enumerate(cleaned):
            if i < len(cleaned) // 2:  # First half - likely letters
                if char.isdigit() and char in corrections.values():
                    # Convert back to letter
                    for letter, digit in corrections.items():
                        if digit == char:
                            corrected += letter
                            break
                    else:
                        corrected += char
                else:
                    corrected += char
            else:  # Second half - likely numbers
                if char.isalpha() and char in corrections:
                    corrected += corrections[char]
                else:
                    corrected += char
        
        return corrected
    
    def get_confidence(self) -> float:
        """Return confidence score of last recognition"""
        return self.last_confidence
