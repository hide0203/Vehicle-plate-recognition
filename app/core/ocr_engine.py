import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
from app.config import Config
import logging

class OCREngine:
    def __init__(self):
        import pytesseract
        from app.config import Config
        pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD

    def preprocess_for_ocr(self, roi):
        import cv2
        import numpy as np
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < 200:
            scale = int(200 / h)
            gray = cv2.resize(gray, (w * scale, 200), interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        if np.mean(thresh) < 127:
            thresh = cv2.bitwise_not(thresh)
        return thresh

    def extract_text(self, roi):
        import pytesseract
        processed_roi = self.preprocess_for_ocr(roi)
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(processed_roi, config=custom_config)
        return text.strip()

    
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
