from mtcnn import MTCNN
import cv2
import numpy as np
from config import MIN_FACE_SIZE

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, frame):
        """Detect faces in a frame and return bounding boxes"""
        # Convert to RGB (MTCNN expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = self.detector.detect_faces(rgb_frame)
        
        # Filter out small faces
        valid_detections = []
        for det in detections:
            x, y, w, h = det['box']
            if w >= MIN_FACE_SIZE[0] and h >= MIN_FACE_SIZE[1]:
                valid_detections.append(det)
        
        return valid_detections

    def extract_face(self, frame, box, margin=20):
        """Extract face region from frame with margin"""
        x, y, w, h = box
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(w + 2 * margin, frame.shape[1] - x)
        h = min(h + 2 * margin, frame.shape[0] - y)
        face = frame[y:y+h, x:x+w]
        return face