from flask import Blueprint, render_template, Response, jsonify, request, flash, redirect, url_for
import cv2
import json
from datetime import datetime, timedelta
import base64
import numpy as np

from app.core.video_processor import video_processor
from app.models.database import db_manager
from app.config import Config

main = Blueprint('main', __name__)

@main.route('/')
def index():
    """Main dashboard page"""
    stats = db_manager.get_statistics()
    recent_plates = db_manager.get_recent_plates(10)
    return render_template('index.html', stats=stats, recent_plates=recent_plates)

@main.route('/dashboard')
def dashboard():
    """Detailed dashboard with analytics"""
    stats = db_manager.get_statistics()
    recent_plates = db_manager.get_recent_plates(50)
    return render_template('dashboard.html', stats=stats, recent_plates=recent_plates)

@main.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            frame = video_processor.get_latest_frame()
            if frame is not None:
                # Encode frame as JPEG
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route('/start_camera')
def start_camera():
    """Start camera processing"""
    if video_processor.start_processing():
        flash('Camera started successfully', 'success')
    else:
        flash('Failed to start camera', 'error')
    return redirect(url_for('main.index'))

@main.route('/stop_camera')
def stop_camera():
    """Stop camera processing"""
    video_processor.stop_processing()
    flash('Camera stopped', 'info')
    return redirect(url_for('main.index'))

@main.route('/search')
def search():
    """Search for plates"""
    query = request.args.get('q', '')
    if query:
        results = db_manager.search_plates(query)
    else:
        results = []
    
    return render_template('search_results.html', results=results, query=query)

@main.route('/api/recent_plates')
def api_recent_plates():
    """API endpoint for recent plates"""
    limit = request.args.get('limit', 10, type=int)
    plates = db_manager.get_recent_plates(limit)
    
    # Convert to JSON serializable format
    result = []
    for plate in plates:
        result.append({
            'id': plate['id'],
            'plate_number': plate['plate_number'],
            'timestamp': plate['timestamp'],
            'confidence_score': plate['confidence_score'],
            'image_path': plate['image_path']
        })
    
    return jsonify(result)

@main.route('/api/statistics')
def api_statistics():
    """API endpoint for statistics"""
    stats = db_manager.get_statistics()
    return jsonify(stats)

@main.route('/api/upload_image', methods=['POST'])
def api_upload_image():
    """API endpoint for single image processing"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image data
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Process image
        results = video_processor.process_single_image_data(frame)
        
        return jsonify({
            'success': True,
            'plates_detected': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
