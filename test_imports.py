"""
Simple test to check if all imports work correctly
"""

print("Testing imports...")

try:
    import cv2
    print("✓ OpenCV imported successfully")
except Exception as e:
    print(f"✗ OpenCV import failed: {e}")

try:
    import mediapipe as mp
    print("✓ MediaPipe imported successfully")
except Exception as e:
    print(f"✗ MediaPipe import failed: {e}")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow imported successfully (version: {tf.__version__})")
except Exception as e:
    print(f"✗ TensorFlow import failed: {e}")

try:
    import face_recognition
    print("✓ face_recognition imported successfully")
except Exception as e:
    print(f"✗ face_recognition import failed: {e}")

try:
    from liveness_detector import LivenessDetector
    print("✓ LivenessDetector imported successfully")
except Exception as e:
    print(f"✗ LivenessDetector import failed: {e}")

try:
    from enhanced_cheat_detection_system import EnhancedCheatDetector
    print("✓ EnhancedCheatDetector imported successfully")
except Exception as e:
    print(f"✗ EnhancedCheatDetector import failed: {e}")

print("\nAll import tests completed.")