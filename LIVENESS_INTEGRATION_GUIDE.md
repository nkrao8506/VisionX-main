# VisionX with Liveness Detection Integration

## Overview
This enhanced VisionX system now includes real/fake face detection (liveness detection) along with the original exercise tracking functionality. The system can detect:

- **Exercise Performance**: Tracks exercise movements and provides feedback
- **Identity Verification**: Verifies if the person matches a registered photo
- **Liveness Detection**: Detects if faces are real or fake (spoofed)
- **Replay Attack Detection**: Detects if someone is showing a video/photo
- **Cheat Prevention**: Comprehensive monitoring to prevent cheating during exercises

## New Features Added

### 1. Liveness Detection Module (`liveness_detector.py`)
- Real-time face detection using DNN face detector
- Liveness classification using TensorFlow/Keras model
- Supports both webcam and video file input
- Provides confidence scores for predictions

### 2. Enhanced Cheat Detection (`enhanced_cheat_detection_system.py`)
- Integrates liveness detection with existing cheat detection
- Tracks fake face detection violations
- Blocks sessions after repeated fake detections
- Maintains all original cheat detection features

### 3. Updated Main System (`main.py`)
- Now uses `EnhancedCheatDetector` instead of `ComprehensiveCheatDetector`
- Displays liveness detection results in real-time
- Shows fake face alerts prominently
- Maintains all exercise tracking functionality

## Files Added/Modified

### New Files:
- `liveness_detector.py` - Core liveness detection functionality
- `enhanced_cheat_detection_system.py` - Enhanced cheat detection with liveness
- `test_integration.py` - Test script for the integration
- `liveness_converted.keras` - Liveness detection AI model
- `label_encoder.pickle` - Label encoder for real/fake classes
- `encoded_faces.pickle` - Face encodings data
- `face_detector/` - DNN face detection model files

### Modified Files:
- `main.py` - Updated to use enhanced cheat detection
- `requirements.txt` - Added TensorFlow and face recognition dependencies

## How It Works

1. **Video Capture**: System captures frames from webcam or video file
2. **Exercise Tracking**: MediaPipe detects body pose and tracks exercise movements
3. **Face Detection**: DNN detector finds faces in each frame
4. **Liveness Check**: AI model determines if detected faces are real or fake
5. **Identity Verification**: Compares faces with registered user photo
6. **Violation Tracking**: Monitors for cheating attempts and fake faces
7. **Session Management**: Blocks session if too many violations detected

## Usage

### Basic Exercise Tracking with Liveness Detection:
```bash
python main.py --exercise_type push-up
```

### Test Liveness Detection Only:
```bash
python test_integration.py
```

### Available Exercise Types:
- `push-up`
- `pull-up`
- `sit-up`
- `squat`
- `vertical jump`
- `run`

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure all model files are in the VisionX-main directory:
   - `liveness_converted.keras`
   - `label_encoder.pickle`
   - `encoded_faces.pickle`
   - `face_detector/deploy.prototxt`
   - `face_detector/res10_300x300_ssd_iter_140000.caffemodel`

## Real-time Feedback

The system now provides:

### Visual Indicators:
- **Green Box**: Real face detected ✓
- **Red Box**: Fake face detected ⚠️
- **Exercise Feedback**: Performance coaching messages
- **Violation Warnings**: Cheating attempt alerts

### Console Output:
- Liveness detection results
- Exercise repetition counts
- Violation notifications
- Session summary statistics

## Violation Types Monitored:

1. **Wrong Person**: Face doesn't match registered user
2. **No Face Detected**: No face visible for extended period
3. **Multiple Faces**: More than one face in frame
4. **Replay Attack**: Video/photo being shown to camera
5. **Fake Face Detected**: AI detects spoofed/artificial face
6. **Poor Lighting**: Insufficient lighting for verification

## Session Blocking:

The system will block the exercise session if:
- Total violations exceed threshold (default: 3)
- Consecutive fake face detections exceed limit (default: 5)
- Extended period without face detection

## Configuration:

Key parameters can be adjusted in the detector classes:
- `face_match_threshold`: Strictness of identity verification
- `liveness_confidence_threshold`: Minimum confidence for liveness
- `max_violations`: Maximum violations before session blocked
- `max_fake_detections`: Maximum fake detections allowed

## Testing:

Run the test script to verify everything works:
```bash
python test_integration.py
```

This will test both standalone liveness detection and the integrated system.

## Troubleshooting:

### Common Issues:
1. **ModuleNotFoundError**: Install missing packages with pip
2. **Model Loading Error**: Ensure all model files are in correct location
3. **Camera Access**: Check webcam permissions and availability
4. **False Positives**: Adjust confidence thresholds if needed

### Performance Notes:
- The system processes video in real-time
- Liveness detection adds minimal processing overhead
- Face detection confidence can be adjusted for accuracy vs speed

## Technical Details:

### Liveness Detection Model:
- Input: 32x32 RGB face images
- Architecture: CNN with batch normalization and dropout
- Output: Binary classification (real/fake)
- Framework: TensorFlow/Keras

### Face Detection:
- Model: OpenCV DNN SSD MobileNet
- Input: 300x300 images
- Confidence threshold: 0.5 (adjustable)

### Integration Architecture:
- Modular design for easy maintenance
- Separate classes for different functionalities
- Event-driven violation tracking
- Comprehensive logging and monitoring