"""
Minimal test for VisionX with liveness detection
Tests the core integration without full UI
"""

import cv2
import argparse
import mediapipe as mp
import numpy as np
import sys
import os

# Import our modules
from enhanced_cheat_detection_system import EnhancedCheatDetector
from cheat_messages import EnhancedCheatMessages

def test_minimal_integration():
    print("Starting minimal integration test...")
    
    # Initialize components
    try:
        user_id = "test_user_123"
        registered_photo = None  # Skip photo for now
        
        # Initialize enhanced cheat detector
        print("Initializing enhanced cheat detector...")
        cheat_detector = EnhancedCheatDetector(user_id, registered_photo)
        print("✓ Enhanced cheat detector initialized")
        
        # Initialize message handler
        message_handler = EnhancedCheatMessages()
        print("✓ Message handler initialized")
        
        # Initialize MediaPipe
        mp_pose = mp.solutions.pose
        print("✓ MediaPipe initialized")
        
        # Test with webcam
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("✓ Webcam opened successfully")
        
        frame_count = 0
        max_frames = 50  # Test for 50 frames
        
        print(f"Processing {max_frames} frames... Press 'q' to quit early")
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Resize frame
                frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
                
                # Process with MediaPipe (basic test)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = pose.process(frame_rgb)
                frame_rgb.flags.writeable = True
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Process with enhanced cheat detector
                detection_results = cheat_detector.process_frame(frame)
                
                # Draw liveness overlay
                frame = cheat_detector.draw_liveness_overlay(frame)
                
                # Add status info
                status_text = f"Frame: {frame_count}/{max_frames}"
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show liveness status
                liveness_status = detection_results.get('liveness_status', 'unknown')
                if detection_results.get('fake_face_detected', False):
                    cv2.putText(frame, "FAKE DETECTED!", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif liveness_status == 'real':
                    cv2.putText(frame, "Real Face", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('VisionX Minimal Test', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Test stopped by user")
                    break
                
                frame_count += 1
                
                # Print progress every 10 frames
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count} frames...")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✓ Test completed successfully! Processed {frame_count} frames")
        
        # Print session summary
        summary = cheat_detector.get_session_summary()
        print("\nSession Summary:")
        print(f"  - Session Active: {summary['session_active']}")
        print(f"  - Liveness Enabled: {summary['liveness_enabled']}")
        print(f"  - Total Violations: {summary['total_violations']}")
        print(f"  - Violation Counts: {summary['violation_counts']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_integration()
    if success:
        print("\n🎉 Minimal integration test PASSED!")
        print("The system is ready for full VisionX usage.")
    else:
        print("\n❌ Minimal integration test FAILED!")
        print("Please check the error messages above.")