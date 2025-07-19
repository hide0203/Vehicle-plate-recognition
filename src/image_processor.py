"""
Image Processing Utilities
Handles image preprocessing, enhancement, and visualization
"""

import cv2
import numpy as np
from typing import List, Tuple

class ImageProcessor:
    def __init__(self, config):
        self.config = config
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better detection"""
        # Resize if too large
        height, width = image.shape[:2]
        max_dimension = self.config['processing']['max_dimension']
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Noise reduction
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        return image
    
    def extract_plate_region(self, image: np.ndarray, 
                           plate_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract license plate region from image"""
        x, y, w, h = plate_coords
        
        # Add padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Extract region
        plate_region = image[y:y+h, x:x+w]
        
        return plate_region
    
    def draw_plates(self, image: np.ndarray, 
                   plates: List[Tuple[int, int, int, int]], 
                   plate_texts: List[str]) -> np.ndarray:
        """Draw bounding boxes and text on detected plates"""
        result = image.copy()
        
        for i, (x, y, w, h) in enumerate(plates):
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw plate text
            if i < len(plate_texts) and plate_texts[i]:
                text = plate_texts[i]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Draw text background
                cv2.rectangle(result, (x, y - text_height - 10), 
                            (x + text_width, y), (0, 255, 0), -1)
                
                # Draw text
                cv2.putText(result, text, (x, y - 5), font, font_scale, 
                          (0, 0, 0), thickness)
        
        return result
    
    def enhance_plate_image(self, plate_image: np.ndarray) -> np.ndarray:
        """Enhance plate image for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(equalized, -1, kernel)
        
        return sharpened
