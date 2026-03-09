# Advanced Posture Detection System

A comprehensive real-time posture monitoring system with eye tracking, hand detection, face mesh, and distance estimation capabilities.

## Features

### 🧍 Posture Detection
- Real-time posture angle measurement
- Customizable good/bad posture thresholds
- Personal posture calibration system
- Continuous monitoring with alerts
- Visual feedback with color-coded status

### 👁️ Eye Tracking & Blink Detection
- Automatic blink counting
- Blinks per minute (BPM) tracking
- Eye strain alerts (< 15 BPM)
- Gaze direction tracking (Left/Right/Center, Up/Down)
- Eye Aspect Ratio (EAR) based detection

### 🖐️ Hand Detection
- Tracks up to 2 hands simultaneously
- Hand skeleton visualization
- Left/Right hand identification
- Real-time hand landmark tracking

### 😊 Face Mesh Detection
- 468 facial landmark tracking
- Face contour visualization
- Iris tracking
- Two modes: Contours (fast) or Full Mesh (detailed)

### 📏 Screen Distance Estimation
- Calibrated distance measurement
- Real-time distance tracking in cm
- Safe distance alerts (20-40 cm recommended)
- "Too Close" and "Too Far" warnings

### 📊 Statistics & Tracking
- Good vs Bad posture time tracking
- Posture score percentage
- Session summaries
- JSON logging capability
- Screenshot capture

## Requirements

```bash
pip install opencv-python>=4.5.0
pip install numpy>=1.19.0
pip install mediapipe==0.10.14
```

## Installation

1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the program:
   ```bash
   python3 posture_advanced.py
   ```

## Usage

### Initial Setup

When you first run the program, you'll be prompted to calibrate:

#### 1. Posture Calibration (Recommended)
- Sit with your BEST upright posture
- Press SPACE to start calibration
- System measures your angle for 30 frames
- Creates personalized good posture range (±15°)

#### 2. Distance Calibration (Recommended)
- Position yourself at comfortable distance
- Press SPACE to start calibration
- Enter your actual distance in cm when prompted
- System tracks distance in real-time

### Controls

| Key | Action |
|-----|--------|
| **Q** or **ESC** | Exit program |
| **SPACE** | Toggle fullscreen mode |
| **S** | Save screenshot |
| **R** | Reset all statistics |
| **C** | Show current configuration |
| **B** | Recalibrate posture baseline |
| **D** | Recalibrate distance |
| **H** | Toggle hand detection on/off |
| **F** | Toggle face detection on/off |
| **M** | Toggle face mesh detail (contours/full) |
| **E** | Toggle eye tracking on/off |
| **X** | Reset blink counter |

## Configuration

Edit the `CONFIG` dictionary at the top of the file to customize:

### Camera Settings
```python
"camera_id": 0,              # 0 for default webcam
"frame_width": 640,
"frame_height": 480,
```

### Posture Settings
```python
"good_posture_min_angle": 140,  # Minimum angle for good posture
"good_posture_max_angle": 180,  # Maximum angle for good posture
"angle_tolerance": 15,          # Tolerance from calibrated angle
```

### Eye Tracking Settings
```python
"blink_threshold": 0.21,        # Eye aspect ratio threshold
"blink_alert_rate": 15,         # Alert if BPM < this value
```

### Distance Settings
```python
"min_safe_distance": 20,        # Minimum safe distance (cm)
"max_safe_distance": 40,        # Maximum safe distance (cm)
```

### Alert Settings
```python
"bad_posture_alert_delay": 10,  # Seconds before alert
"enable_visual_alert": True,    # Flash screen on bad posture
```

## Display Information

### On-Screen Display

**Top Left:**
- Good posture time (seconds)
- Bad posture time (seconds)
- Posture score percentage

**Top Right:**
- Hands detected count
- Face detected indicator
- Blink statistics (total blinks, BPM)
- Eye strain alert
- Distance measurement
- Distance status (Good/Too Close/Too Far)

**Center:**
- Main posture status (Good/Bad)
- Current angle measurement
- Gaze direction (when looking)
- Hand labels (Left/Right)

**Bottom:**
- Control hints

## Understanding the Measurements

### Posture Angle
- Measured between: Nose → Shoulder → Hip
- **Good Range:** 140-180° (default, adjustable)
- **< 140°:** Slouching forward
- **> 180°:** Leaning back

### Blink Rate
- **Normal:** 15-20 blinks per minute
- **< 15 BPM:** Eye strain alert (blink more!)
- **Tracked:** Total blinks and average BPM

### Screen Distance
- **Optimal:** 20-40 cm (8-16 inches)
- **< 20 cm:** Too close (eye strain risk)
- **> 40 cm:** Too far (may strain to see)

## Tips for Best Results

### Posture Detection
1. Ensure your upper body (head to hips) is visible
2. Sit facing the camera directly
3. Good lighting helps detection
4. Calibrate for personalized thresholds

### Eye Tracking
1. Look at the camera occasionally
2. Blink naturally (15-20 times/minute)
3. Take breaks if eye strain alert appears
4. Follow 20-20-20 rule: Every 20 min, look 20 feet away for 20 sec

### Distance Estimation
1. Calibrate at your preferred working distance
2. Maintain consistent distance during work
3. Adjust if "Too Close" warnings appear
4. Recommended: 30 cm for laptops, 50-70 cm for monitors

### Hand Detection
1. Keep hands in frame for tracking
2. Works best with good lighting
3. Can track both hands simultaneously

### Face Mesh
1. Face camera directly for best results
2. Use "Contours" mode for better performance
3. Switch to "Full Mesh" for detailed visualization

## Performance Optimization

If experiencing lag:

1. **Disable unused features:**
   - Press **H** to disable hands
   - Press **F** to disable face
   - Press **E** to disable eyes

2. **Use lighter face mesh:**
   - Press **M** to switch to contours mode

3. **Reduce resolution:**
   ```python
   "frame_width": 320,
   "frame_height": 240,
   ```

4. **Lower detection confidence:**
   ```python
   "detection_confidence": 0.3,
   ```

## Output Files

### Screenshots
- Format: `posture_screenshot_YYYYMMDD_HHMMSS.jpg`
- Saved in current directory
- Press **S** to capture

### Logs (if enabled)
- File: `posture_log.json`
- Contains: timestamp, status, angle, times
- Logged every 60 seconds (configurable)

## Troubleshooting

### Camera Not Opening
```bash
# Check camera permissions
ls /dev/video*

# Try different camera ID
"camera_id": 1,  # or 2, 3, etc.
```

### No Person Detected
- Move back to show head to hips
- Improve lighting
- Check camera is not blocked
- Lower detection confidence to 0.3

### Pose Detected But Not Tracked
- Ensure shoulders and hips are visible
- Move back from camera
- Check lighting conditions
- Visibility needs to be > 10%

### Eye Tracking Not Working
- Face the camera directly
- Ensure face is well-lit
- Eyes must be clearly visible
- Try adjusting `blink_threshold`

### Distance Estimation Inaccurate
- Recalibrate distance (press **D**)
- Ensure face is fully visible
- Measure actual distance accurately
- Face camera directly

## Health & Ergonomics

### Recommended Posture
- Sit upright with back straight
- Shoulders relaxed and back
- Head aligned over shoulders
- Feet flat on floor
- Arms at 90° angle

### Screen Position
- Top of screen at or below eye level
- Distance: 20-40 cm (arm's length)
- Screen perpendicular to windows (reduce glare)
- Slight downward gaze (15-20°)

### Break Recommendations
- **20-20-20 Rule:** Every 20 min, look 20 feet away for 20 sec
- **Hourly:** Stand and stretch for 5 minutes
- **Blink:** Consciously blink 15-20 times per minute
- **Posture:** Adjust position every 30 minutes

## Technical Details

### Pose Detection
- **Library:** MediaPipe Pose
- **Landmarks:** 33 body keypoints
- **Key Points:** Nose (0), Shoulders (11,12), Hips (23,24)
- **Algorithm:** BlazePose

### Eye Tracking
- **Method:** Eye Aspect Ratio (EAR)
- **Landmarks:** 6 points per eye
- **Blink Threshold:** EAR < 0.21
- **Iris Tracking:** MediaPipe iris landmarks

### Hand Detection
- **Library:** MediaPipe Hands
- **Landmarks:** 21 points per hand
- **Max Hands:** 2 (configurable)

### Face Mesh
- **Landmarks:** 468 facial points
- **Iris:** Included with refine_landmarks
- **Modes:** Contours (fast) or Tesselation (detailed)

### Distance Estimation
- **Method:** Face width measurement
- **Formula:** distance = (calibrated_distance × calibrated_width) / current_width
- **Accuracy:** ±5 cm (depends on calibration)

## Session Summary

At program exit, you'll see:

```
=== Session Summary ===
Good Posture Time: 450 seconds
Bad Posture Time: 150 seconds
Good Posture Percentage: 75.0%

Eye Tracking Summary:
Total Blinks: 120
Average Blinks Per Minute: 18
Recommended: 15-20 blinks per minute

Distance Summary:
Last Measured Distance: 32 cm
Recommended Range: 20-40 cm (8-16 inches)
======================
```

## Credits

- **MediaPipe:** Google's ML solutions for pose, hands, and face
- **OpenCV:** Computer vision library
- **NumPy:** Numerical computing

## License

This project is provided as-is for educational and personal use.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify all dependencies are installed
3. Ensure camera permissions are granted
4. Try adjusting configuration values

---

**Stay healthy! Maintain good posture! 🧍‍♂️💪**
