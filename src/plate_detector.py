"""
License Plate Detection Module
Uses Haar cascades and contour detection for plate localization
"""

import cv2
import numpy as np
from typing import List, Tuple

class PlateDetector:
    def __init__(self, config):
        self.config = config
        self.cascade = cv2.CascadeClassifier(config['models']['cascade_path'])
        self.min_plate_area = config['detection']['min_plate_area']
        self.max_plate_area = config['detection']['max_plate_area']
        
    def detect_plates(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect license plates in the image
        Returns list of (x, y, width, height) tuples
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect using Haar cascade
        cascade_plates = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 30),
            maxSize=(400, 150)
        )
        
        # Additional contour-based detection
        contour_plates = self._detect_by_contours(gray)
        
        # Combine and filter results
        all_plates = list(cascade_plates) + contour_plates
        filtered_plates = self._filter_plates(all_plates)
        
        return filtered_plates
    
    def _detect_by_contours(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect plates using contour analysis"""
        plates = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio and area
            aspect_ratio = w / h
            area = w * h
            
            if (2.0 <= aspect_ratio <= 6.0 and 
                self.min_plate_area <= area <= self.max_plate_area):
                plates.append((x, y, w, h))
        
        return plates
    
    def _filter_plates(self, plates: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Filter out duplicate and invalid detections"""
        if not plates:
            return []
        
        # Remove duplicates using Non-Maximum Suppression
        plates_array = np.array(plates)
        
        # Calculate overlap scores
        keep_indices = []
        for i, plate in enumerate(plates):
            x1, y1, w1, h1 = plate
            overlap = False
            
            for j, other_plate in enumerate(plates):
                if i != j:
                    x2, y2, w2, h2 = other_plate
                    
                    # Calculate intersection over union
                    intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * \
                                      max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    
                    union_area = w1 * h1 + w2 * h2 - intersection_area
                    
                    if intersection_area / union_area > 0.3:  # 30% overlap threshold
                        overlap = True
                        break
            
            if not overlap:
                keep_indices.append(i)
        
        return [plates[i] for i in keep_indices]
