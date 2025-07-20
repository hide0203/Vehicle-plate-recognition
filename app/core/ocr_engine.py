import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
from app.config import Config
import logging

class OCREngine:
    def __init__(self):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Set Tesseract command path
        pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
        
        # OCR configuration for license plates
        self.config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    def preprocess_for_ocr(self, image):
    # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Sharpen image (optional)
        # kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        # gray = cv2.filter2D(gray, -1, kernel)

        # Resize for better OCR (important!)
        height, width = gray.shape
        if height < 100 or width < 200:
            scale_factor = max(2, int(200 / min(height, width)))
            gray = cv2.resize(gray, (width * scale_factor, height * scale_factor))

        # Increase contrast and threshold
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    
    def extract_text(self, image):
        """Extract text from plate image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_for_ocr(image)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(processed_image)
            
            # Perform OCR
            raw_text = pytesseract.image_to_string(pil_image, config=self.config)
            
            # Clean and validate text
            cleaned_text = self.clean_text(raw_text)
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return ""
    
    def clean_text(self, raw_text):
        """Clean and validate extracted text"""
        # Remove whitespace and newlines
        text = raw_text.strip().replace('\n', '').replace('\r', '')
        
        # Remove special characters except alphanumeric
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Basic validation for plate format
        if self.is_valid_plate(text):
            return text
        
        return ""
    
    def is_valid_plate(self, text):
        """Validate if text looks like a license plate"""
        if not text or len(text) < 4 or len(text) > 10:
            return False
        
        # Check if it contains both letters and numbers
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        return has_letter and has_number
    
    def get_confidence_score(self, image):
        """Get OCR confidence score"""
        try:
            processed_image = self.preprocess_for_ocr(image)
            pil_image = Image.fromarray(processed_image)
            
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            
            if confidences:
                return sum(confidences) / len(confidences)
            return 0
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return 0
