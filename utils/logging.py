import csv
from datetime import datetime
from config import LOG_FILE

def init_log_file():
    """Initialize log file with headers if it doesn't exist"""
    try:
        with open(LOG_FILE, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'status', 'person', 'confidence', 'image_path'])
    except FileExistsError:
        pass

def log_recognition(timestamp, status, person=None, confidence=None, image_path=None):
    """Log a recognition event"""
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, status, person, confidence, image_path])