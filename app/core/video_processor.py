import cv2
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
import logging

from app.core.plate_detector import PlateDetector
from app.core.ocr_engine import OCREngine
from app.models.database import db_manager
from app.config import Config

class VideoProcessor:
    def __init__(self):
        self.detector = PlateDetector()
        self.ocr_engine = OCREngine()
        self.logger = logging.getLogger(__name__)
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()
        self.is_running = False
        
        # Camera
        self.cap = None
        
        # Processing settings
        self.process_every_n_frames = 5  # Process every 5th frame for performance
        self.frame_counter = 0
        
    def initialize_camera(self, camera_index=0):
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, Config.FPS)
            
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
                
            return True
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False
    
    def start_processing(self):
        """Start video processing threads"""
        if not self.initialize_camera(Config.CAMERA_INDEX):
            return False
        
        self.is_running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.process_thread.start()
        
        return True
    
    def stop_processing(self):
        """Stop video processing"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def _capture_frames(self):
        """Capture frames from camera"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Only add frame if queue is not full
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
            else:
                time.sleep(0.01)
    
    def _process_frames(self):
        """Process frames for plate detection"""
        while self.is_running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=1)
                    self.frame_counter += 1
                    
                    # Process every nth frame for performance
                    if self.frame_counter % self.process_every_n_frames == 0:
                        self._detect_and_store_plates(frame)
                    
                    # Put frame in result queue for display
                    if not self.result_queue.full():
                        self.result_queue.put(frame)
                        
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Frame processing error: {e}")
    
    def _detect_and_store_plates(self, frame):
        """Detect plates and store results"""
        try:
            # Detect plates
            plates = self.detector.detect_plates(frame)
            
            for plate_coords in plates:
                x, y, w, h = plate_coords

                plate_roi = self.detector.extract_plate_roi(frame, plate_coords)
                print(f"[DEBUG] Extracted ROI shape: {plate_roi.shape}")

                if plate_roi.size == 0:
                    print("[DEBUG] ROI size zero, skipping this region.")
                
                # Extract plate ROI
                plate_roi = self.detector.extract_plate_roi(frame, plate_coords)
                
                if plate_roi.size == 0:
                    continue
                
                # Perform OCR
                plate_text = self.ocr_engine.extract_text(plate_roi)
                print(f"[DEBUG] OCR text: '{plate_text}'")
                
                if plate_text:
                    # Get confidence score
                    confidence = self.ocr_engine.get_confidence_score(plate_roi)
                    
                    # Save plate image
                    timestamp = datetime.now()
                    image_filename = f"plate_{timestamp.strftime('%Y%m%d_%H%M%S')}_{plate_text}.jpg"
                    image_path = Config.CAPTURED_PLATES_DIR / image_filename
                    
                    cv2.imwrite(str(image_path), plate_roi)
                    
                    # Store in database
                    db_manager.insert_plate(
                        plate_number=plate_text,
                        confidence_score=confidence,
                        image_path=str(image_filename)
                    )
                    
                    self.logger.info(f"Detected plate: {plate_text} (confidence: {confidence:.2f})")
                    
        except Exception as e:
            self.logger.error(f"Plate detection error: {e}")
    
    def get_latest_frame(self):
        """Get latest processed frame"""
        try:
            if not self.result_queue.empty():
                return self.result_queue.get_nowait()
        except queue.Empty:
            pass
        return None
    
    def process_single_image(self, image_path):
        """Process a single image file"""
        try:
            frame = cv2.imread(str(image_path))
            if frame is None:
                return []
            
            plates = self.detector.detect_plates(frame)
            results = []
            
            for plate_coords in plates:
                plate_roi = self.detector.extract_plate_roi(frame, plate_coords)
                plate_text = self.ocr_engine.extract_text(plate_roi)
                confidence = self.ocr_engine.get_confidence_score(plate_roi)
                
                if plate_text:
                    results.append({
                        'plate_number': plate_text,
                        'confidence': confidence,
                        'coordinates': plate_coords
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Single image processing error: {e}")
            return []

# Global video processor instance
video_processor = VideoProcessor()
