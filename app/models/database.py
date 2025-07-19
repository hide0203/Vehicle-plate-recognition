import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from app.config import Config

class DatabaseManager:
    def __init__(self):
        self.db_path = Config.DATABASE_PATH
        self._local = threading.local()
        self.init_database()
    
    def get_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(str(self.db_path))
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create plates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detected_plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                confidence_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                vehicle_type TEXT,
                location TEXT,
                status TEXT DEFAULT 'detected'
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_plate_number 
            ON detected_plates(plate_number)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON detected_plates(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_plate(self, plate_number, confidence_score=None, 
                    image_path=None, vehicle_type=None, location=None):
        """Insert detected plate into database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detected_plates 
            (plate_number, confidence_score, image_path, vehicle_type, location)
            VALUES (?, ?, ?, ?, ?)
        ''', (plate_number, confidence_score, image_path, vehicle_type, location))
        
        conn.commit()
        return cursor.lastrowid
    
    def get_recent_plates(self, limit=50):
        """Get recently detected plates"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM detected_plates 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        return cursor.fetchall()
    
    def search_plates(self, plate_number):
        """Search for specific plate number"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM detected_plates 
            WHERE plate_number LIKE ? 
            ORDER BY timestamp DESC
        ''', (f'%{plate_number}%',))
        
        return cursor.fetchall()
    
    def get_statistics(self):
        """Get database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total plates detected
        cursor.execute('SELECT COUNT(*) FROM detected_plates')
        total_plates = cursor.fetchone()[0]
        
        # Unique plates
        cursor.execute('SELECT COUNT(DISTINCT plate_number) FROM detected_plates')
        unique_plates = cursor.fetchone()[0]
        
        # Today's detections
        cursor.execute('''
            SELECT COUNT(*) FROM detected_plates 
            WHERE DATE(timestamp) = DATE('now')
        ''')
        today_detections = cursor.fetchone()[0]
        
        return {
            'total_plates': total_plates,
            'unique_plates': unique_plates,
            'today_detections': today_detections
        }

# Global database instance
db_manager = DatabaseManager()
