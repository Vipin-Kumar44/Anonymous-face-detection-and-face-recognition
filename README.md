# Anonymous-face-detection-and-face-recognition
![DEMO image](https://example.com/myimage.png)

## 🚀 Key Features
- **High-Speed Processing**: Optimized for 25 FPS on edge devices
- **Smart Resource Management**: Automatic resolution scaling & frame skipping
- **Dual-Mode Recognition**: 
  - Standard mode (high accuracy) 
  - Fast mode (30% faster with slight accuracy trade-off)
- **GPU Acceleration**: Supports CUDA, TensorRT, and OpenVINO
- **Privacy-First**: 100% offline processing

## ⚡ Performance Optimization Techniques
1. Dynamic resolution scaling (480p ↔ 720p)
2. Intelligent frame skipping
3. MTCNN parameter tuning
4. FaceNet batch processing
5. GPU-accelerated inference
6. Quantized TensorFlow Lite models

## Note:-
-working on python 3.9
-downlode facenet_keras.h5 and store it in models -----https://www.kaggle.com/datasets/utkarshsaxenadn/facenet-keras?utm_source=chatgpt.com

## 📦 Installation
```bash
# Install with performance extras
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118  # For CUDA support


