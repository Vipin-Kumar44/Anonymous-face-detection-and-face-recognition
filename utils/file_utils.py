import os
import cv2
from datetime import datetime
from config import UNKNOWN_FACES_DIR

def ensure_dir_exists(directory):
    """Ensure directory exists, create if not"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# def save_unknown_face(face_image):
#     """Save detected unknown face with timestamp"""
#     ensure_dir_exists(UNKNOWN_FACES_DIR)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#     filename = f"unknown_{timestamp}.jpg"
#     filepath = os.path.join(UNKNOWN_FACES_DIR, filename)
#     cv2.imwrite(filepath, face_image)
#     return filepath