"""
Face Liveness Detection Module for VisionX
Integrates real/fake face detection with exercise tracking system
"""

import cv2
import numpy as np
import tensorflow as tf
import pickle
import os
from typing import Tuple, Dict, Optional


class LivenessDetector:
    """
    Face liveness detection system that determines if a face is real or fake (spoofed).
    Integrates with VisionX exercise tracking system.
    """
    
    def __init__(self, model_path: str = 'liveness_converted.keras', 
                 label_encoder_path: str = 'label_encoder.pickle',
                 detector_folder: str = 'face_detector',
                 confidence_threshold: float = 0.5):
        """
        Initialize the liveness detector
        
        Args:
            model_path: Path to the liveness detection model
            label_encoder_path: Path to the label encoder pickle file
            detector_folder: Path to face detector model files
            confidence_threshold: Minimum confidence for face detection
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.label_encoder = None
        self.face_detector = None
        
        # Load components
        self._load_liveness_model(model_path)
        self._load_label_encoder(label_encoder_path)
        self._load_face_detector(detector_folder)
        
        print("[INFO] Liveness detector initialized successfully")
    
    def _load_liveness_model(self, model_path: str) -> None:
        """Load the TensorFlow/Keras liveness detection model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"[INFO] Liveness model loaded from {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load liveness model: {e}")
            raise
    
    def _load_label_encoder(self, label_encoder_path: str) -> None:
        """Load the label encoder for class names"""
        try:
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"[INFO] Label encoder loaded from {label_encoder_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load label encoder: {e}")
            raise
    
    def _load_face_detector(self, detector_folder: str) -> None:
        """Load the DNN face detector"""
        try:
            proto_path = os.path.join(detector_folder, 'deploy.prototxt')
            model_path = os.path.join(detector_folder, 'res10_300x300_ssd_iter_140000.caffemodel')
            
            if not os.path.exists(proto_path) or not os.path.exists(model_path):
                raise FileNotFoundError(f"Face detector files not found in {detector_folder}")
            
            self.face_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            print(f"[INFO] Face detector loaded from {detector_folder}")
        except Exception as e:
            print(f"[ERROR] Failed to load face detector: {e}")
            raise
    
    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Detect faces in the frame using DNN face detector
        
        Args:
            frame: Input image frame
            
        Returns:
            List of face bounding boxes [(startX, startY, endX, endY), ...]
        """
        if self.face_detector is None:
            return []
        
        try:
            # Get frame dimensions
            (h, w) = frame.shape[:2]
            
            # Create blob from frame
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300), 
                (104.0, 177.0, 123.0)
            )
            
            # Pass blob through network
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()
            
            faces = []
            
            # Process detections
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > self.confidence_threshold:
                    # Get bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype('int')
                    
                    # Expand bounding box slightly and ensure it's within frame
                    startX = max(0, startX - 20)
                    startY = max(0, startY - 20)
                    endX = min(w, endX + 20)
                    endY = min(h, endY + 20)
                    
                    faces.append((startX, startY, endX, endY))
            
            return faces
            
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
            return []
    
    def predict_liveness(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """
        Predict if a face ROI is real or fake
        
        Args:
            face_roi: Face region of interest (cropped face image)
            
        Returns:
            Tuple of (prediction_label, confidence_score)
        """
        if self.model is None or self.label_encoder is None:
            return "unknown", 0.0
        
        try:
            # Preprocess the face
            face = cv2.resize(face_roi, (32, 32))
            face = face.astype('float') / 255.0
            face = tf.keras.preprocessing.image.img_to_array(face)
            face = np.expand_dims(face, axis=0)
            
            # Make prediction
            predictions = self.model.predict(face, verbose=0)[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(predictions)
            label_name = self.label_encoder.classes_[predicted_class_idx]
            confidence = predictions[predicted_class_idx]
            
            return label_name, float(confidence)
            
        except Exception as e:
            print(f"[ERROR] Liveness prediction failed: {e}")
            return "unknown", 0.0
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a complete frame for liveness detection
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary containing detection results:
            {
                'faces_detected': int,
                'liveness_results': [{'bbox': tuple, 'label': str, 'confidence': float}, ...],
                'has_fake_face': bool,
                'all_faces_real': bool
            }
        """
        results = {
            'faces_detected': 0,
            'liveness_results': [],
            'has_fake_face': False,
            'all_faces_real': False
        }
        
        # Detect faces in frame
        faces = self.detect_faces(frame)
        results['faces_detected'] = len(faces)
        
        if len(faces) == 0:
            return results
        
        # Process each detected face
        real_faces = 0
        fake_faces = 0
        
        for face_bbox in faces:
            startX, startY, endX, endY = face_bbox
            
            # Extract face ROI
            face_roi = frame[startY:endY, startX:endX]
            
            if face_roi.size == 0:
                continue
            
            # Predict liveness
            label, confidence = self.predict_liveness(face_roi)
            
            # Store result
            result = {
                'bbox': face_bbox,
                'label': label,
                'confidence': confidence
            }
            results['liveness_results'].append(result)
            
            # Count real vs fake faces
            if label == 'real':
                real_faces += 1
            elif label == 'fake':
                fake_faces += 1
        
        # Set flags
        results['has_fake_face'] = fake_faces > 0
        results['all_faces_real'] = real_faces > 0 and fake_faces == 0
        
        return results
    
    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw liveness detection results on the frame
        
        Args:
            frame: Input frame
            results: Results from process_frame()
            
        Returns:
            Frame with annotations
        """
        annotated_frame = frame.copy()
        
        for result in results['liveness_results']:
            startX, startY, endX, endY = result['bbox']
            label = result['label']
            confidence = result['confidence']
            
            # Choose color based on prediction
            if label == 'real':
                color = (0, 255, 0)  # Green for real
                text_color = (0, 255, 0)
            elif label == 'fake':
                color = (0, 0, 255)  # Red for fake
                text_color = (0, 0, 255)
            else:
                color = (0, 255, 255)  # Yellow for unknown
                text_color = (0, 255, 255)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (startX, startY), (endX, endY), color, 2)
            
            # Draw label and confidence
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(annotated_frame, label_text, (startX, startY - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Add fake alert if detected
            if label == 'fake':
                cv2.putText(annotated_frame, "FAKE DETECTED!", (startX, endY + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated_frame


def main():
    """Test the liveness detector"""
    detector = LivenessDetector()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results = detector.process_frame(frame)
        
        # Draw results
        annotated_frame = detector.draw_results(frame, results)
        
        # Display info
        info_text = f"Faces: {results['faces_detected']}, Fake: {results['has_fake_face']}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Liveness Detection Test', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()