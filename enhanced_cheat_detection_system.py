# enhanced_cheat_detection_system.py
import cv2
import numpy as np
import face_recognition
import pickle
import time
from datetime import datetime
import json
import os
from typing import Tuple, Dict, List
import logging

# Import our liveness detector
from liveness_detector import LivenessDetector


class EnhancedCheatDetector:
    """
    Enhanced cheat detection system with integrated liveness detection
    Combines identity verification, replay attack detection, and real/fake face detection
    """
    
    def __init__(self, user_id: str, registered_photo_path: str = None):
        self.user_id = user_id
        self.registered_encoding = None
        self.session_log = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Detection models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize liveness detector
        try:
            self.liveness_detector = LivenessDetector()
            self.liveness_enabled = True
            self.logger.info("Liveness detection enabled")
        except Exception as e:
            self.logger.error(f"Failed to initialize liveness detector: {e}")
            self.liveness_detector = None
            self.liveness_enabled = False
        
        # Thresholds and counters
        self.face_match_threshold = 0.6
        self.replay_confidence_threshold = 0.7
        self.liveness_confidence_threshold = 0.7  # Minimum confidence for liveness
        self.max_violations = 3
        self.max_no_face_frames = 90  # 3 seconds at 30fps
        self.max_fake_detections = 5  # Max fake detections before blocking
        
        # Violation tracking
        self.violation_counts = {
            'wrong_person': 0,
            'no_face_detected': 0,
            'multiple_faces': 0,
            'replay_detected': 0,
            'poor_lighting': 0,
            'fake_face_detected': 0  # New violation type
        }
        
        # Session state
        self.session_active = True
        self.last_face_detection = time.time()
        self.no_face_frame_count = 0
        self.consecutive_fake_detections = 0
        
        # Face verification tracking (only verify once at beginning)
        self.face_verified_once = False
        self.initial_verification_frames = 90  # Check face for first 3 seconds (30fps)
        self.frame_count = 0
        
        # Load registered face if provided
        if registered_photo_path and os.path.exists(registered_photo_path):
            self.load_registered_face(registered_photo_path)
    
    def load_registered_face(self, photo_path: str) -> bool:
        """Load and encode the registered user's photo"""
        try:
            # Load image
            image = face_recognition.load_image_file(photo_path)
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 0:
                self.logger.error("No face found in registered photo")
                return False
            elif len(encodings) > 1:
                self.logger.warning("Multiple faces found, using the first one")
            
            self.registered_encoding = encodings[0]
            
            # Save encoding for future use
            encoding_file = f"user_encodings/{self.user_id}_encoding.pkl"
            os.makedirs("user_encodings", exist_ok=True)
            with open(encoding_file, 'wb') as f:
                pickle.dump(self.registered_encoding, f)
            
            self.logger.info(f"Successfully loaded registered face for user {self.user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading registered face: {str(e)}")
            return False
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """Detect all faces in the current frame"""
        faces = []
        
        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using face_recognition (more accurate)
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                faces.append({
                    'location': (top, right, bottom, left),
                    'encoding': encoding,
                    'center': ((left + right) // 2, (top + bottom) // 2),
                    'size': (right - left) * (bottom - top)
                })
        
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
        
        return faces
    
    def verify_identity(self, current_face_encoding: np.ndarray) -> Tuple[bool, float]:
        """Verify if the current face matches the registered user"""
        if self.registered_encoding is None:
            return False, 0.0
        
        # Compare faces
        distance = face_recognition.face_distance([self.registered_encoding], current_face_encoding)[0]
        is_match = distance <= self.face_match_threshold
        confidence = 1 - distance  # Convert distance to confidence score
        
        return is_match, confidence
    
    def detect_replay_attack(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Detect if the video is a replay/screen recording"""
        try:
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            replay_indicators = []
            
            # 1. Screen reflection detection
            bright_pixels = np.sum(gray > 240) / gray.size
            replay_indicators.append(bright_pixels > 0.15)  # Too many bright pixels
            
            # 2. Edge sharpness (screens have artificial sharp edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / edges.size
            replay_indicators.append(edge_density > 0.08)  # Too many sharp edges
            
            # 3. Color temperature analysis (screens have blue tint)
            b, g, r = cv2.split(frame)
            blue_dominance = np.mean(b) / (np.mean(r) + np.mean(g) + np.mean(b) + 1e-6)
            replay_indicators.append(blue_dominance > 0.4)  # Too much blue
            
            # Calculate replay confidence
            replay_score = sum(replay_indicators) / len(replay_indicators)
            is_replay = replay_score > self.replay_confidence_threshold
            
            return is_replay, replay_score
            
        except Exception as e:
            self.logger.error(f"Replay detection failed: {e}")
            return False, 0.0
    
    def process_liveness_detection(self, frame: np.ndarray) -> Dict:
        """Process liveness detection on the frame"""
        if not self.liveness_enabled or self.liveness_detector is None:
            return {
                'liveness_enabled': False,
                'faces_detected': 0,
                'liveness_results': [],
                'has_fake_face': False,
                'all_faces_real': False
            }
        
        try:
            # Use our liveness detector
            results = self.liveness_detector.process_frame(frame)
            results['liveness_enabled'] = True
            return results
            
        except Exception as e:
            self.logger.error(f"Liveness detection failed: {e}")
            return {
                'liveness_enabled': False,
                'faces_detected': 0,
                'liveness_results': [],
                'has_fake_face': False,
                'all_faces_real': False
            }
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a frame for comprehensive cheat detection including liveness
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary with detection results and session status
        """
        current_time = time.time()
        self.frame_count += 1
        
        # Initialize result structure
        result = {
            'session_active': self.session_active,
            'face_verified': self.face_verified_once,
            'violations': [],
            'warnings': [],
            'overlay_color': (255, 255, 255),  # White default
            'liveness_status': 'unknown',
            'identity_verified': self.face_verified_once,
            'replay_detected': False,
            'fake_face_detected': False
        }
        
        if not self.session_active:
            return result
        
        # 1. LIVENESS DETECTION (NEW)
        liveness_results = self.process_liveness_detection(frame)
        
        if liveness_results['liveness_enabled']:
            result['liveness_status'] = 'processed'
            
            if liveness_results['has_fake_face']:
                self.consecutive_fake_detections += 1
                self.violation_counts['fake_face_detected'] += 1
                result['fake_face_detected'] = True
                result['violations'].append("Fake face detected!")
                result['overlay_color'] = (0, 0, 255)  # Red
                
                # Block session if too many fake detections
                if self.consecutive_fake_detections >= self.max_fake_detections:
                    self.session_active = False
                    result['session_active'] = False
                    self.logger.warning(f"Session blocked due to repeated fake face detection")
                    return result
            
            elif liveness_results['all_faces_real'] and liveness_results['faces_detected'] > 0:
                self.consecutive_fake_detections = 0  # Reset counter
                result['liveness_status'] = 'real'
                result['overlay_color'] = (0, 255, 0)  # Green
        
        # 2. IDENTITY VERIFICATION (Only during initial frames)
        if not self.face_verified_once and self.frame_count <= self.initial_verification_frames:
            faces = self.detect_faces_in_frame(frame)
            
            if len(faces) == 0:
                self.no_face_frame_count += 1
                result['warnings'].append("No face detected - initial verification needed")
                
                if self.no_face_frame_count > self.max_no_face_frames:
                    self.violation_counts['no_face_detected'] += 1
                    result['violations'].append("Face required for initial verification")
                    
            elif len(faces) > 1:
                self.violation_counts['multiple_faces'] += 1
                result['violations'].append("Multiple faces detected during verification")
                result['overlay_color'] = (0, 165, 255)  # Orange
                
            else:
                # Single face detected during initial verification
                self.no_face_frame_count = 0
                self.last_face_detection = current_time
                
                face = faces[0]
                
                # Verify identity only once
                if self.registered_encoding is not None:
                    is_match, confidence = self.verify_identity(face['encoding'])
                    
                    if is_match:
                        self.face_verified_once = True  # Set the flag
                        result['face_verified'] = True
                        result['identity_verified'] = True
                        if result['overlay_color'] == (255, 255, 255):  # Only if not set by liveness
                            result['overlay_color'] = (0, 255, 0)  # Green
                        self.logger.info(f"Face verified successfully at frame {self.frame_count}")
                    else:
                        self.violation_counts['wrong_person'] += 1
                        result['violations'].append(f"Identity verification failed (confidence: {confidence:.2f})")
                        result['overlay_color'] = (0, 0, 255)  # Red
        
        elif self.face_verified_once:
            # Face already verified, skip identity checks but keep result updated
            result['face_verified'] = True
            result['identity_verified'] = True
        
        elif self.frame_count > self.initial_verification_frames and not self.face_verified_once:
            # Verification period ended without successful verification
            result['violations'].append("Initial face verification failed - session blocked")
            self.session_active = False
            result['session_active'] = False
        
        # 3. REPLAY DETECTION (always active throughout video)
        is_replay, replay_confidence = self.detect_replay_attack(frame)
        if is_replay:
            self.violation_counts['replay_detected'] += 1
            result['violations'].append(f"Screen/replay attack detected (confidence: {replay_confidence:.2f})")
            result['replay_detected'] = True
            result['overlay_color'] = (0, 0, 255)  # Red
        
        # 4. CHECK VIOLATION LIMITS
        total_violations = sum(self.violation_counts.values())
        if total_violations >= self.max_violations:
            self.session_active = False
            result['session_active'] = False
            self.logger.warning(f"Session blocked due to {total_violations} violations")
        
        # 5. LOG EVENTS
        if result['violations']:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'violations': result['violations'],
                'violation_counts': self.violation_counts.copy()
            }
            self.session_log.append(log_entry)
        
        return result
    
    def draw_liveness_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw liveness detection overlay on frame"""
        if not self.liveness_enabled or self.liveness_detector is None:
            return frame
        
        try:
            # Process liveness and get results
            liveness_results = self.liveness_detector.process_frame(frame)
            
            # Draw liveness results
            annotated_frame = self.liveness_detector.draw_results(frame, liveness_results)
            
            return annotated_frame
            
        except Exception as e:
            self.logger.error(f"Failed to draw liveness overlay: {e}")
            return frame
    
    def get_session_summary(self) -> Dict:
        """Get summary of the detection session"""
        return {
            'user_id': self.user_id,
            'session_active': self.session_active,
            'violation_counts': self.violation_counts.copy(),
            'total_violations': sum(self.violation_counts.values()),
            'session_log': self.session_log.copy(),
            'liveness_enabled': self.liveness_enabled
        }