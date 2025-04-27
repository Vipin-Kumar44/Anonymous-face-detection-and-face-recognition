import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import cv2
import time
from datetime import datetime
from config import FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from utils.logging import init_log_file, log_recognition

def main():
    # Initialize components
    init_log_file()
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    # Open video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Frame timing for FPS control
    frame_time = 1 / TARGET_FPS
    last_time = time.time()
    
    # Counter for periodic cleanup
    frame_count = 0
    CLEANUP_INTERVAL = 1800  # Cleanup every 1800 frames (~1 minute at 30 FPS)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Control FPS
        current_time = time.time()
        elapsed = current_time - last_time
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)
        last_time = current_time
        
        # Detect faces
        detections = detector.detect_faces(frame)
        
        for det in detections:
            x, y, w, h = det['box']
            
            # Extract face
            face = detector.extract_face(frame, det['box'])
            
            # Recognize face
            embedding = recognizer.get_embedding(face)
            person, confidence = recognizer.recognize_face(embedding, face)

            # Draw rectangle and label
            color = (0, 255, 0) if person else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            if person:
                label = f"{person} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                log_recognition(datetime.now(), "recognized", person, confidence)
            else:
                log_recognition(datetime.now(), "unknown", None, confidence)
        
        # Display frame
        cv2.imshow('Face Recognition', frame)
        
        # Periodic cleanup of old unknown faces
        frame_count += 1
        if frame_count % CLEANUP_INTERVAL == 0:
            recognizer.cleanup_old_unknown_faces()
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()