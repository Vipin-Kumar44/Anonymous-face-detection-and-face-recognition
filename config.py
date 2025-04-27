import os

# Directory configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, 'known_faces')
UNKNOWN_FACES_DIR = os.path.join(BASE_DIR, 'unknown_faces')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model configurations
FACENET_MODEL_PATH = "D:/anonymous_face_ detection/models/facenet_keras.h5"
# FACENET_MODEL_PATH_JSON = "model/facenet_keras.json"
# FACENET_MODEL_PATH_WEIGHTS = "model/facenet_keras_weights.h5"
# RECOGNITION_THRESHOLD = 0.7  # or adjust as needed



# Recognition settings
RECOGNITION_THRESHOLD = 0.6  # Cosine similarity threshold
MIN_FACE_SIZE = (40, 40)     # Minimum face size to process
UNKNOWN_SIMILARITY_THRESHOLD = 0.85  # Threshold for considering faces duplicates

# Performance settings
TARGET_FPS = 25
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Logging
LOG_FILE = os.path.join(BASE_DIR, 'recognition_logs.csv')