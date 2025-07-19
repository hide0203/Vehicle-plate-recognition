#!/usr/bin/env python3
"""
Vehicle Number Plate Recognition System
Main application entry point
"""

import os
import sys
import argparse
import cv2
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from plate_detector import PlateDetector
from plate_recognizer import PlateRecognizer
from image_processor import ImageProcessor
from utils import load_config, save_results

def main():
    """Main application function"""
    parser = argparse.ArgumentParser(description='Vehicle Number Plate Recognition System')
    parser.add_argument('--input', '-i', required=True, help='Input image or video path')
    parser.add_argument('--output', '-o', default='data/output', help='Output directory')
    parser.add_argument('--config', '-c', default='config/config.yaml', help='Configuration file')
    parser.add_argument('--mode', '-m', choices=['image', 'video'], default='image', help='Processing mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize components
    detector = PlateDetector(config)
    recognizer = PlateRecognizer(config)
    processor = ImageProcessor(config)
    
    # Process input
    if args.mode == 'image':
        process_image(args.input, args.output, detector, recognizer, processor)
    else:
        process_video(args.input, args.output, detector, recognizer, processor)

def process_image(input_path, output_dir, detector, recognizer, processor):
    """Process a single image"""
    print(f"Processing image: {input_path}")
    
    # Load and preprocess image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not load image {input_path}")
        return
    
    processed_image = processor.preprocess(image)
    
    # Detect license plates
    plates = detector.detect_plates(processed_image)
    
    results = []
    for i, plate_region in enumerate(plates):
        # Extract plate region
        plate_image = processor.extract_plate_region(image, plate_region)
        
        # Recognize text
        plate_text = recognizer.recognize_text(plate_image)
        
        # Store results
        result = {
            'plate_number': plate_text,
            'confidence': recognizer.get_confidence(),
            'coordinates': plate_region,
            'image_path': input_path
        }
        results.append(result)
        
        # Save detected plate image
        plate_filename = f"plate_{i+1}.jpg"
        plate_path = os.path.join(output_dir, 'detected_plates', plate_filename)
        cv2.imwrite(plate_path, plate_image)
        
        print(f"Detected plate {i+1}: {plate_text}")
    
    # Save results
    save_results(results, output_dir)
    
    # Draw bounding boxes and save annotated image
    annotated_image = processor.draw_plates(image, plates, [r['plate_number'] for r in results])
    output_path = os.path.join(output_dir, 'annotated_' + os.path.basename(input_path))
    cv2.imwrite(output_path, annotated_image)
    
    print(f"Results saved to: {output_dir}")

def process_video(input_path, output_dir, detector, recognizer, processor):
    """Process video file"""
    print(f"Processing video: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video writer
    output_path = os.path.join(output_dir, 'processed_' + os.path.basename(input_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    all_results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 10th frame for performance
        if frame_count % 10 == 0:
            processed_frame = processor.preprocess(frame)
            plates = detector.detect_plates(processed_frame)
            
            frame_results = []
            for plate_region in plates:
                plate_image = processor.extract_plate_region(frame, plate_region)
                plate_text = recognizer.recognize_text(plate_image)
                
                frame_results.append({
                    'frame': frame_count,
                    'plate_number': plate_text,
                    'confidence': recognizer.get_confidence(),
                    'coordinates': plate_region
                })
            
            all_results.extend(frame_results)
            
            # Draw annotations
            annotated_frame = processor.draw_plates(frame, plates, 
                                                  [r['plate_number'] for r in frame_results])
        else:
            annotated_frame = frame
        
        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    cap.release()
    out.release()
    
    # Save all results
    save_results(all_results, output_dir)
    print(f"Video processing complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
