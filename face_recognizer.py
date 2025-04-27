import os
import numpy as np
import cv2
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime
from collections import defaultdict
from config import KNOWN_FACES_DIR, RECOGNITION_THRESHOLD, UNKNOWN_FACES_DIR, UNKNOWN_SIMILARITY_THRESHOLD
from utils.file_utils import ensure_dir_exists

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FaceRecognizer:
    def __init__(self):
        # Load FaceNet model using keras-facenet wrapper
        self.embedder = self.load_facenet_model()

        # Initialize known embeddings dictionary
        self.known_embeddings = {}

        # Store unknown face embeddings with timestamps
        self.unknown_embeddings = []

        # Load known face embeddings
        self.load_known_faces()

        # Load existing unknown faces to prevent duplicates
        self.load_existing_unknown_faces()

    def load_facenet_model(self):
        try:
            return FaceNet()
        except Exception as e:
            raise RuntimeError(f"Failed to load FaceNet model: {e}")

    def preprocess_input(self, img):
        """Resize and convert image to RGB"""
        img = cv2.resize(img, (160, 160))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

    def get_embedding(self, face_img):
        """Get face embedding from FaceNet model"""
        face_img = self.preprocess_input(face_img)
        embeddings = self.embedder.embeddings([face_img])
        return embeddings[0]

    def load_known_faces(self):
        """Load known faces from directory and compute embeddings"""
        ensure_dir_exists(KNOWN_FACES_DIR)

        for person_name in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue

            self.known_embeddings[person_name] = []

            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    embedding = self.get_embedding(img)
                    self.known_embeddings[person_name].append(embedding)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    def load_existing_unknown_faces(self):
        """Load embeddings of existing unknown faces to prevent duplicates"""
        ensure_dir_exists(UNKNOWN_FACES_DIR)
        
        for img_name in os.listdir(UNKNOWN_FACES_DIR):
            img_path = os.path.join(UNKNOWN_FACES_DIR, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                embedding = self.get_embedding(img)
                timestamp = os.path.getmtime(img_path)
                self.unknown_embeddings.append((embedding, timestamp, img_path))
            except Exception as e:
                print(f"Error loading unknown face {img_path}: {e}")

    def is_new_unknown_face(self, embedding):
        """Check if this unknown face hasn't been seen before"""
        if not self.unknown_embeddings:
            return True
            
        # Convert to numpy array for batch processing
        stored_embeddings = np.array([e[0] for e in self.unknown_embeddings])
        
        # Calculate all similarities at once
        similarities = cosine_similarity([embedding], stored_embeddings)[0]
        
        # Return True only if all similarities are below threshold
        return not any(sim > UNKNOWN_SIMILARITY_THRESHOLD for sim in similarities)

    def save_unknown_face(self, face_img):
        """Save unknown face only if it's truly new"""
        embedding = self.get_embedding(face_img)
        
        if not self.is_new_unknown_face(embedding):
            print("Skipping duplicate unknown face")
            return None
            
        # Generate a unique filename using datetime format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_name = f"unknown_{timestamp}.jpg"
        img_path = os.path.join(UNKNOWN_FACES_DIR, img_name)
        
        # Save the image
        try:
            cv2.imwrite(img_path, face_img)
            self.unknown_embeddings.append((embedding, time.time(), img_path))
            print(f"New unknown face saved to {img_path}")
            return img_path
        except Exception as e:
            print(f"Failed to save unknown face: {e}")
            return None

    def recognize_face(self, face_embedding, face_img):
        """Compare face embedding with known faces and return best match"""
        best_match = None
        highest_similarity = 0

        for person_name, embeddings in self.known_embeddings.items():
            for known_embedding in embeddings:
                similarity = cosine_similarity(
                    [face_embedding],
                    [known_embedding]
                )[0][0]

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = person_name

        if highest_similarity > RECOGNITION_THRESHOLD:
            return best_match, highest_similarity
        else:
            # Handle unknown face with deduplication
            face_path = self.save_unknown_face(face_img)
            return None, highest_similarity

    def cleanup_old_unknown_faces(self, max_age_days=30):
        """Remove unknown faces older than specified days"""
        now = time.time()
        removed = 0
        
        for i in range(len(self.unknown_embeddings)-1, -1, -1):
            _, timestamp, filepath = self.unknown_embeddings[i]
            age_days = (now - timestamp) / (24 * 3600)
            
            if age_days > max_age_days:
                try:
                    os.remove(filepath)
                    self.unknown_embeddings.pop(i)
                    removed += 1
                    print(f"Removed old unknown face: {filepath}")
                except Exception as e:
                    print(f"Failed to remove old unknown face {filepath}: {e}")
        
        print(f"Removed {removed} old unknown faces")
        return removed