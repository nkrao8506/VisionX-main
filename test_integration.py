"""
Test script for the integrated VisionX system with liveness detection
Run this to test the liveness detection integration before running the full exercise system
"""

import cv2
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from liveness_detector import LivenessDetector
from enhanced_cheat_detection_system import EnhancedCheatDetector


def test_liveness_detector():
    """Test the standalone liveness detector"""
    print("Testing standalone liveness detector...")
    
    try:
        detector = LivenessDetector()
        print("✓ Liveness detector initialized successfully")
        
        # Test with webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return False
        
        print("Press 'q' to quit the liveness test")
        
        frame_count = 0
        while frame_count < 100:  # Test for 100 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = detector.process_frame(frame)
            
            # Draw results
            annotated_frame = detector.draw_results(frame, results)
            
            # Display info
            info_text = f"Faces: {results['faces_detected']}"
            if results['has_fake_face']:
                info_text += " | FAKE DETECTED!"
            elif results['all_faces_real']:
                info_text += " | All Real"
            
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Liveness Detection Test', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Liveness detector test completed")
        return True
        
    except Exception as e:
        print(f"✗ Liveness detector test failed: {e}")
        return False


def test_enhanced_cheat_detector():
    """Test the enhanced cheat detector with liveness"""
    print("\nTesting enhanced cheat detector...")
    
    try:
        # Use a dummy registered photo path (will work without it)
        detector = EnhancedCheatDetector("test_user", None)
        print("✓ Enhanced cheat detector initialized successfully")
        
        # Test with webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return False
        
        print("Press 'q' to quit the enhanced cheat detection test")
        
        frame_count = 0
        while frame_count < 100:  # Test for 100 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = detector.process_frame(frame)
            
            # Draw liveness overlay
            frame = detector.draw_liveness_overlay(frame)
            
            # Display detection results
            info_lines = [
                f"Session Active: {results['session_active']}",
                f"Liveness: {results.get('liveness_status', 'unknown')}",
                f"Fake Detected: {results.get('fake_face_detected', False)}"
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 30 + (i * 25)
                cv2.putText(frame, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show violations
            if results.get('violations'):
                violation_text = " | ".join(results['violations'])
                cv2.putText(frame, f"Violations: {violation_text}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.imshow('Enhanced Cheat Detection Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        summary = detector.get_session_summary()
        print("Session Summary:")
        print(f"  - Liveness Enabled: {summary['liveness_enabled']}")
        print(f"  - Total Violations: {summary['total_violations']}")
        print(f"  - Violation Counts: {summary['violation_counts']}")
        
        print("✓ Enhanced cheat detector test completed")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced cheat detector test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("VisionX Liveness Detection Integration Test")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        'liveness_converted.keras',
        'label_encoder.pickle',
        'face_detector/deploy.prototxt',
        'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
    ]
    
    print("Checking required files...")
    all_files_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n✗ Some required files are missing. Please ensure all model files are copied to VisionX-main folder.")
        return
    
    print("\n" + "=" * 50)
    
    # Test liveness detector
    liveness_test_passed = test_liveness_detector()
    
    # Test enhanced cheat detector
    enhanced_test_passed = test_enhanced_cheat_detector()
    
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print(f"Liveness Detector: {'✓ PASSED' if liveness_test_passed else '✗ FAILED'}")
    print(f"Enhanced Cheat Detector: {'✓ PASSED' if enhanced_test_passed else '✗ FAILED'}")
    
    if liveness_test_passed and enhanced_test_passed:
        print("\n🎉 All tests passed! The integration is ready.")
        print("\nYou can now run the full VisionX system with:")
        print("python main.py --exercise_type push-up")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main()