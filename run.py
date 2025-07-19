import sys
from pathlib import Path
import os

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask
from app.config import Config
from app.web.routes import main
from app.models.database import db_manager

def create_app():
    """Create and configure Flask application"""
    # Get absolute path to templates directory
    base_dir = Path(__file__).parent
    template_dir = base_dir / "app" / "web" / "templates"
    static_dir = base_dir / "app" / "static"
    
    app = Flask(__name__, 
                template_folder=str(template_dir),
                static_folder=str(static_dir))
    app.config.from_object(Config)
    
    # Register blueprints
    app.register_blueprint(main)
    
    return app

if __name__ == '__main__':
    try:
        Config.create_directories()
        db_manager.init_database()
        
        app = create_app()
        
        print(f"Starting Vehicle Plate Recognition System...")
        print(f"Template directory: {Path(__file__).parent / 'app' / 'web' / 'templates'}")
        print(f"Web interface available at http://{Config.HOST}:{Config.PORT}")
        
        app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, threaded=True)
        
    except Exception as e:
        print(f"Error: {e}")
