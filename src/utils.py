"""
Utility functions for the VNPR system
"""

import yaml
import json
import os
from datetime import datetime
from typing import Dict, List, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        # Return default configuration
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Return default configuration"""
    return {
        'models': {
            'cascade_path': 'models/haarcascade_license_plate.xml'
        },
        'detection': {
            'min_plate_area': 1000,
            'max_plate_area': 50000
        },
        'recognition': {
            'confidence_threshold': 0.6
        },
        'processing': {
            'max_dimension': 1200
        },
        'use_gpu': False
    }

def save_results(results: List[Dict], output_dir: str):
    """Save detection results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp
    timestamp = datetime.now().isoformat()
    output_data = {
        'timestamp': timestamp,
        'total_detections': len(results),
        'results': results
    }
    
    # Save to JSON
    output_path = os.path.join(output_dir, 'results.json')
    with open(output_path, 'w') as file:
        json.dump(output_data, file, indent=2)
    
    # Save to CSV for easy viewing
    csv_path = os.path.join(output_dir, 'results.csv')
    with open(csv_path, 'w') as file:
        file.write('Plate Number,Confidence,Coordinates,Image Path\n')
        for result in results:
            coords = result.get('coordinates', [])
            coords_str = f"({','.join(map(str, coords))})" if coords else ""
            file.write(f"{result.get('plate_number', '')},{result.get('confidence', 0)},{coords_str},{result.get('image_path', '')}\n")

def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        'src',
        'models',
        'data/input/sample_images',
        'data/input/test_videos',
        'data/output/detected_plates',
        'data/output/results',
        'tests',
        'docs',
        'config'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
def validate_image_formats(file_path: str) -> bool:
    """Validate if file is a supported image format"""
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    return any(file_path.lower().endswith(fmt) for fmt in supported_formats)

def validate_video_formats(file_path: str) -> bool:
    """Validate if file is a supported video format"""
    supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    return any(file_path.lower().endswith(fmt) for fmt in supported_formats)
