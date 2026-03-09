import cv2
import mediapipe as mp
import math
import time
import json
from datetime import datetime

# ==================== CONFIGURATION ====================
CONFIG = {
    # Camera Settings
    "camera_id": 0,  # 0 for default webcam, 1 for external camera
    "frame_width": 640,
    "frame_height": 480,
    
    # Posture Detection Settings
    "good_posture_min_angle": 140,  # Minimum angle for good posture (lowered)
    "good_posture_max_angle": 180,  # Maximum angle for good posture (increased)
    "angle_tolerance": 15,          # Degrees of tolerance from calibrated angle
    "detection_confidence": 0.5,    # 0.0 to 1.0
    "tracking_confidence": 0.5,     # 0.0 to 1.0
    "use_calibration": False,       # Use calibrated baseline angle
    "calibrated_angle": None,       # Will be set during calibration
    
    # Hand Detection Settings
    "enable_hand_detection": True,  # Enable hand tracking
    "max_hands": 2,                 # Maximum number of hands to detect
    "hand_detection_confidence": 0.5,
    "hand_tracking_confidence": 0.5,
    "show_hand_landmarks": True,    # Show hand skeleton
    
    # Face Detection Settings
    "enable_face_detection": True,  # Enable face mesh tracking
    "max_faces": 1,                 # Maximum number of faces to detect
    "face_detection_confidence": 0.5,
    "face_tracking_confidence": 0.5,
    "show_face_mesh": True,         # Show face landmarks (skeleton)
    "face_mesh_tesselation": False, # Show mesh connections (disable for performance)
    
    # Eye Tracking Settings
    "enable_eye_tracking": True,    # Enable eye tracking and blink detection
    "show_eye_gaze": True,          # Show gaze direction indicators
    "blink_threshold": 0.21,        # Eye aspect ratio threshold for blink detection
    "blink_alert_rate": 15,         # Alert if blinks per minute < this value
    "track_blink_stats": True,      # Track blink statistics
    
    # Distance Estimation Settings
    "enable_distance_estimation": True,  # Enable screen distance estimation
    "calibrated_distance": None,    # Calibrated reference distance (cm)
    "calibrated_face_width": None,  # Face width at calibrated distance
    "min_safe_distance": 20,        # Minimum safe distance in cm (8 inches)
    "max_safe_distance": 40,        # Maximum safe distance in cm (16 inches)
    "show_distance_warning": True,  # Show warning when too close/far
    
    # Alert Settings
    "bad_posture_alert_delay": 10,  # Seconds of bad posture before alert
    "enable_visual_alert": True,    # Flash screen on bad posture
    "enable_sound_alert": False,    # Beep sound (requires additional setup)
    
    # Display Settings
    "show_skeleton": True,          # Show pose landmarks
    "show_angles": True,            # Show angle measurements
    "show_statistics": True,        # Show time statistics
    "mirror_mode": True,            # Flip frame horizontally
    
    # Logging Settings
    "enable_logging": True,         # Save posture data to file
    "log_interval": 60,             # Log every N seconds
    "log_file": "posture_log.json"
}

# ==================== INITIALIZATION ====================
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=CONFIG["detection_confidence"],
    min_tracking_confidence=CONFIG["tracking_confidence"]
)
mp_draw = mp.solutions.drawing_utils

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=CONFIG["max_hands"],
    min_detection_confidence=CONFIG["hand_detection_confidence"],
    min_tracking_confidence=CONFIG["hand_tracking_confidence"]
)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=CONFIG["max_faces"],
    refine_landmarks=True,  # Include iris landmarks
    min_detection_confidence=CONFIG["face_detection_confidence"],
    min_tracking_confidence=CONFIG["face_tracking_confidence"]
)

# Hand landmark drawing specs
hand_drawing_spec = mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
hand_connection_spec = mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2)

# Face landmark drawing specs
face_drawing_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
face_connection_spec = mp_draw.DrawingSpec(color=(255, 255, 0), thickness=1)

# Initialize webcam
cap = cv2.VideoCapture(CONFIG["camera_id"])

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])

# Variables for tracking
bad_posture_time = 0
good_posture_time = 0
last_time = time.time()
bad_posture_continuous = 0
alert_triggered = False
last_log_time = time.time()
posture_log = []

# Eye tracking variables
blink_counter = 0
total_blinks = 0
blink_start_time = time.time()
eye_closed_frames = 0
blink_detected = False
blinks_per_minute = 0
last_blink_time = time.time()
eye_strain_alert = False

# Distance estimation variables
estimated_distance = 0
distance_alert = False
distance_status = "Unknown"

# ==================== FUNCTIONS ====================
def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return abs(angle)

def save_log():
    """Save posture log to file"""
    if CONFIG["enable_logging"] and posture_log:
        try:
            with open(CONFIG["log_file"], 'w') as f:
                json.dump(posture_log, f, indent=2)
            print(f"Log saved to {CONFIG['log_file']}")
        except Exception as e:
            print(f"Error saving log: {e}")

def add_log_entry(status, angle):
    """Add entry to posture log"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "angle": angle,
        "good_time": int(good_posture_time),
        "bad_time": int(bad_posture_time)
    }
    posture_log.append(entry)

def draw_text_with_background(frame, text, position, font_scale=0.6, 
                              text_color=(255, 255, 255), bg_color=(0, 0, 0), 
                              thickness=2):
    """Draw text with background for better visibility"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(frame, (x - 5, y - text_h - 5), 
                 (x + text_w + 5, y + 5), bg_color, -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

def calculate_eye_aspect_ratio(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    # Vertical eye landmarks
    A = math.sqrt((eye_landmarks[1][0] - eye_landmarks[5][0])**2 + 
                  (eye_landmarks[1][1] - eye_landmarks[5][1])**2)
    B = math.sqrt((eye_landmarks[2][0] - eye_landmarks[4][0])**2 + 
                  (eye_landmarks[2][1] - eye_landmarks[4][1])**2)
    
    # Horizontal eye landmark
    C = math.sqrt((eye_landmarks[0][0] - eye_landmarks[3][0])**2 + 
                  (eye_landmarks[0][1] - eye_landmarks[3][1])**2)
    
    # Eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def get_eye_landmarks(face_landmarks, eye_indices, w, h):
    """Extract eye landmark coordinates"""
    landmarks = face_landmarks.landmark
    eye_points = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        eye_points.append((x, y))
    return eye_points

def calculate_gaze_direction(face_landmarks, w, h):
    """
    Calculate approximate gaze direction based on iris position
    Returns: (horizontal_direction, vertical_direction)
    """
    landmarks = face_landmarks.landmark
    
    # Left eye iris center (landmark 468)
    left_iris = landmarks[468]
    # Right eye iris center (landmark 473)
    right_iris = landmarks[473]
    
    # Left eye corners
    left_eye_left = landmarks[33]
    left_eye_right = landmarks[133]
    
    # Right eye corners
    right_eye_left = landmarks[362]
    right_eye_right = landmarks[263]
    
    # Calculate iris position relative to eye corners
    left_ratio = (left_iris.x - left_eye_left.x) / (left_eye_right.x - left_eye_left.x)
    right_ratio = (right_iris.x - right_eye_left.x) / (right_eye_right.x - right_eye_left.x)
    
    avg_ratio = (left_ratio + right_ratio) / 2
    
    # Determine gaze direction
    if avg_ratio < 0.4:
        h_direction = "Left"
    elif avg_ratio > 0.6:
        h_direction = "Right"
    else:
        h_direction = "Center"
    
    # Vertical gaze (simplified)
    avg_iris_y = (left_iris.y + right_iris.y) / 2
    avg_eye_y = (left_eye_left.y + left_eye_right.y + right_eye_left.y + right_eye_right.y) / 4
    
    if avg_iris_y < avg_eye_y - 0.01:
        v_direction = "Up"
    elif avg_iris_y > avg_eye_y + 0.01:
        v_direction = "Down"
    else:
        v_direction = "Center"
    
    return h_direction, v_direction

def calculate_face_width(face_landmarks, w):
    """
    Calculate face width in pixels (distance between left and right face edges)
    """
    landmarks = face_landmarks.landmark
    
    # Left face edge (landmark 234)
    left_face = landmarks[234]
    # Right face edge (landmark 454)
    right_face = landmarks[454]
    
    face_width_pixels = abs(right_face.x - left_face.x) * w
    
    return face_width_pixels

def estimate_distance(face_width_pixels, calibrated_width, calibrated_distance):
    """
    Estimate distance from camera using face width
    Formula: distance = (calibrated_distance * calibrated_width) / current_width
    """
    if calibrated_width is None or calibrated_distance is None:
        return None
    
    if face_width_pixels == 0:
        return None
    
    estimated_distance = (calibrated_distance * calibrated_width) / face_width_pixels
    return estimated_distance

def calibrate_distance(cap, pose, face_mesh):
    """Calibrate distance by measuring face width at known distance"""
    print("\n=== DISTANCE CALIBRATION ===")
    print("Position yourself at a comfortable distance from the screen.")
    print("Recommended: 20-40 cm (8-16 inches)")
    print("Press SPACE when ready to calibrate...")
    
    face_widths = []
    calibrating = False
    calibration_frames = 0
    required_frames = 30
    
    # Create fullscreen window for calibration
    calib_window = "Distance Calibration"
    cv2.namedWindow(calib_window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(calib_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        if CONFIG["mirror_mode"]:
            frame = cv2.flip(frame, 1)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        h, w, _ = frame.shape
        
        if face_results and face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # Draw face mesh
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(255, 255, 0), thickness=1)
            )
            
            face_width = calculate_face_width(face_landmarks, w)
            
            if calibrating:
                face_widths.append(face_width)
                calibration_frames += 1
                
                progress = (calibration_frames / required_frames) * 100
                draw_text_with_background(frame, f"Calibrating... {int(progress)}%", 
                                         (50, 50), font_scale=1.2, 
                                         text_color=(0, 255, 255), bg_color=(0, 100, 100), thickness=3)
                
                if calibration_frames >= required_frames:
                    avg_face_width = sum(face_widths) / len(face_widths)
                    
                    # Ask user for actual distance
                    cv2.destroyAllWindows()
                    print(f"\nMeasured face width: {avg_face_width:.1f} pixels")
                    
                    while True:
                        try:
                            distance_input = input("Enter your current distance from screen in cm (e.g., 60): ")
                            actual_distance = float(distance_input)
                            if actual_distance > 0:
                                break
                            else:
                                print("Please enter a positive number.")
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                    
                    CONFIG["calibrated_distance"] = actual_distance
                    CONFIG["calibrated_face_width"] = avg_face_width
                    
                    print(f"\n✓ Distance calibration complete!")
                    print(f"  Reference distance: {actual_distance} cm")
                    print(f"  Reference face width: {avg_face_width:.1f} pixels")
                    print("  Starting distance tracking...\n")
                    
                    cv2.destroyWindow(calib_window)
                    time.sleep(1)
                    return
            else:
                draw_text_with_background(frame, "Position at comfortable distance", 
                                         (50, 50), font_scale=1, 
                                         text_color=(0, 255, 0), bg_color=(0, 100, 0), thickness=2)
                draw_text_with_background(frame, f"Face width: {int(face_width)} px", 
                                         (50, 100), font_scale=0.8, 
                                         text_color=(255, 255, 255), bg_color=(0, 0, 0))
                draw_text_with_background(frame, "Press SPACE to calibrate", 
                                         (50, 150), font_scale=0.8, 
                                         text_color=(255, 255, 0), bg_color=(50, 50, 0))
        else:
            draw_text_with_background(frame, "Position your face in frame", 
                                     (50, 50), font_scale=1, 
                                     text_color=(0, 255, 255), bg_color=(0, 100, 100))
        
        cv2.imshow(calib_window, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            if face_results and face_results.multi_face_landmarks:
                calibrating = True
                face_widths = []
                calibration_frames = 0
        elif key == 27 or key == ord('q'):  # ESC or Q to skip
            print("Distance calibration skipped.")
            cv2.destroyWindow(calib_window)
            return

def calibrate_posture(cap, pose):
    """Calibrate good posture by measuring current angle"""
    print("\n=== POSTURE CALIBRATION ===")
    print("Sit in your BEST, most comfortable upright posture.")
    print("Press SPACE when ready to calibrate...")
    
    angles = []
    calibrating = False
    calibration_frames = 0
    required_frames = 30  # Collect 30 frames for average
    
    # Create fullscreen window for calibration
    calib_window = "Posture Calibration"
    cv2.namedWindow(calib_window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(calib_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        if CONFIG["mirror_mode"]:
            frame = cv2.flip(frame, 1)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        h, w, _ = frame.shape
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            shoulder = [landmarks[11].x * w, landmarks[11].y * h]
            ear = [landmarks[7].x * w, landmarks[7].y * h]
            hip = [landmarks[23].x * w, landmarks[23].y * h]
            
            angle = calculate_angle(ear, shoulder, hip)
            
            # Draw skeleton
            mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Draw key points
            cv2.circle(frame, (int(ear[0]), int(ear[1])), 8, (255, 0, 0), -1)
            cv2.circle(frame, (int(shoulder[0]), int(shoulder[1])), 8, (255, 0, 0), -1)
            cv2.circle(frame, (int(hip[0]), int(hip[1])), 8, (255, 0, 0), -1)
            
            if calibrating:
                angles.append(angle)
                calibration_frames += 1
                
                progress = (calibration_frames / required_frames) * 100
                draw_text_with_background(frame, f"Calibrating... {int(progress)}%", 
                                         (50, 50), font_scale=1.2, 
                                         text_color=(0, 255, 255), bg_color=(0, 100, 100), thickness=3)
                
                if calibration_frames >= required_frames:
                    avg_angle = sum(angles) / len(angles)
                    CONFIG["calibrated_angle"] = avg_angle
                    CONFIG["use_calibration"] = True
                    CONFIG["good_posture_min_angle"] = avg_angle - CONFIG["angle_tolerance"]
                    CONFIG["good_posture_max_angle"] = avg_angle + CONFIG["angle_tolerance"]
                    
                    print(f"\n✓ Calibration complete!")
                    print(f"  Your good posture angle: {avg_angle:.1f}°")
                    print(f"  Acceptable range: {CONFIG['good_posture_min_angle']:.1f}° - {CONFIG['good_posture_max_angle']:.1f}°")
                    print("  Starting posture detection...\n")
                    
                    cv2.destroyWindow(calib_window)
                    time.sleep(2)
                    return
            else:
                draw_text_with_background(frame, "Sit with GOOD POSTURE", 
                                         (50, 50), font_scale=1.2, 
                                         text_color=(0, 255, 0), bg_color=(0, 100, 0), thickness=3)
                draw_text_with_background(frame, f"Current angle: {int(angle)}°", 
                                         (50, 100), font_scale=0.8, 
                                         text_color=(255, 255, 255), bg_color=(0, 0, 0))
                draw_text_with_background(frame, "Press SPACE to calibrate", 
                                         (50, 150), font_scale=0.8, 
                                         text_color=(255, 255, 0), bg_color=(50, 50, 0))
        else:
            draw_text_with_background(frame, "Position yourself in frame", 
                                     (50, 50), font_scale=1, 
                                     text_color=(0, 255, 255), bg_color=(0, 100, 100))
        
        cv2.imshow(calib_window, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            if result.pose_landmarks:
                calibrating = True
                angles = []
                calibration_frames = 0
        elif key == 27 or key == ord('q'):  # ESC or Q to skip
            print("Calibration skipped. Using default angles.")
            cv2.destroyWindow(calib_window)
            return

# ==================== MAIN LOOP ====================
print("Advanced Posture Detection Started!")
print("\nWould you like to calibrate your posture? (Recommended)")
print("  Y - Calibrate (sit with good posture and press SPACE)")
print("  N - Use default angles (140-180°)")

calibrate_choice = input("Your choice (Y/N): ").strip().upper()

if calibrate_choice == 'Y':
    calibrate_posture(cap, pose)
else:
    print(f"Using default angle range: {CONFIG['good_posture_min_angle']}-{CONFIG['good_posture_max_angle']}°")

# Distance calibration
if CONFIG["enable_distance_estimation"]:
    print("\nWould you like to calibrate screen distance? (Recommended)")
    print("  Y - Calibrate (measure your comfortable viewing distance)")
    print("  N - Skip (distance estimation will be less accurate)")
    
    distance_choice = input("Your choice (Y/N): ").strip().upper()
    
    if distance_choice == 'Y':
        calibrate_distance(cap, pose, face_mesh)
    else:
        print("Distance calibration skipped.")

print("\n=== Controls ===")
print("Q or ESC: Exit")
print("S: Save screenshot")
print("R: Reset statistics")
print("C: Show configuration")
print("B: Recalibrate posture baseline")
print("D: Recalibrate distance")
print("H: Toggle hand detection")
print("F: Toggle face detection")
print("M: Toggle face mesh detail (contours/full)")
print("E: Toggle eye tracking")
print("X: Reset blink counter")
print("SPACE: Toggle fullscreen")
print("================\n")

# Create window and set to fullscreen
window_name = "Advanced Posture Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
fullscreen = True

try:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Mirror mode
        if CONFIG["mirror_mode"]:
            frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for pose
        result = pose.process(rgb)
        
        # Process the frame for hands
        hand_results = None
        if CONFIG["enable_hand_detection"]:
            hand_results = hands.process(rgb)
        
        # Process the frame for face
        face_results = None
        if CONFIG["enable_face_detection"]:
            face_results = face_mesh.process(rgb)
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Track time
        current_time = time.time()
        elapsed = current_time - last_time
        last_time = current_time
        
        if result.pose_landmarks:
            # Draw skeleton if enabled
            if CONFIG["show_skeleton"]:
                mp_draw.draw_landmarks(
                    frame, 
                    result.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            
            landmarks = result.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[11].x * w, landmarks[11].y * h]
            ear = [landmarks[7].x * w, landmarks[7].y * h]
            hip = [landmarks[23].x * w, landmarks[23].y * h]
            
            # Calculate posture angle
            angle = calculate_angle(ear, shoulder, hip)
            
            # Draw key points
            cv2.circle(frame, (int(ear[0]), int(ear[1])), 8, (255, 0, 0), -1)
            cv2.circle(frame, (int(shoulder[0]), int(shoulder[1])), 8, (255, 0, 0), -1)
            cv2.circle(frame, (int(hip[0]), int(hip[1])), 8, (255, 0, 0), -1)
            
            # Determine posture status
            if CONFIG["good_posture_min_angle"] <= angle <= CONFIG["good_posture_max_angle"]:
                # Good posture
                good_posture_time += elapsed
                bad_posture_continuous = 0
                alert_triggered = False
                
                status_text = "Good Posture"
                status_color = (0, 255, 0)
                bg_color = (0, 100, 0)
            else:
                # Bad posture
                bad_posture_time += elapsed
                bad_posture_continuous += elapsed
                
                # Determine if leaning forward or backward
                if angle < CONFIG["good_posture_min_angle"]:
                    posture_type = "Slouching Forward"
                else:
                    posture_type = "Leaning Back"
                
                # Check if alert should be triggered
                if bad_posture_continuous >= CONFIG["bad_posture_alert_delay"]:
                    if not alert_triggered:
                        alert_triggered = True
                        print(f"⚠️  ALERT: {posture_type} for {int(bad_posture_continuous)} seconds!")
                    
                    status_text = f"BAD POSTURE - {posture_type}!"
                    
                    # Visual alert - flash red border
                    if CONFIG["enable_visual_alert"]:
                        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)
                else:
                    status_text = f"Bad Posture - {posture_type}"
                
                status_color = (0, 0, 255)
                bg_color = (0, 0, 100)
            
            # Display main status
            draw_text_with_background(frame, status_text, (50, 50), 
                                     font_scale=1.2, text_color=status_color, 
                                     bg_color=bg_color, thickness=3)
            
            # Display angle if enabled
            if CONFIG["show_angles"]:
                angle_text = f"Angle: {int(angle)}°"
                draw_text_with_background(frame, angle_text, (10, h - 20),
                                         font_scale=0.6, text_color=(255, 255, 255),
                                         bg_color=(0, 0, 0))
        else:
            # No pose detected
            draw_text_with_background(frame, "No person detected", (50, 50),
                                     font_scale=1, text_color=(0, 255, 255),
                                     bg_color=(0, 100, 100))
        
        # Display statistics if enabled
        if CONFIG["show_statistics"]:
            stats_y = 30
            draw_text_with_background(frame, f"Good: {int(good_posture_time)}s", 
                                     (10, stats_y), font_scale=0.6, 
                                     text_color=(0, 255, 0), bg_color=(0, 50, 0))
            
            draw_text_with_background(frame, f"Bad: {int(bad_posture_time)}s", 
                                     (10, stats_y + 30), font_scale=0.6,
                                     text_color=(0, 0, 255), bg_color=(50, 0, 0))
            
            if good_posture_time + bad_posture_time > 0:
                percentage = (good_posture_time / (good_posture_time + bad_posture_time)) * 100
                draw_text_with_background(frame, f"Score: {percentage:.1f}%", 
                                         (10, stats_y + 60), font_scale=0.6,
                                         text_color=(255, 255, 0), bg_color=(50, 50, 0))
        
        # Process and display hand detection
        if CONFIG["enable_hand_detection"] and hand_results and hand_results.multi_hand_landmarks:
            for hand_idx, (hand_landmarks, handedness) in enumerate(zip(
                hand_results.multi_hand_landmarks, 
                hand_results.multi_handedness
            )):
                # Draw hand landmarks
                if CONFIG["show_hand_landmarks"]:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        hand_drawing_spec,
                        hand_connection_spec
                    )
                
                # Get hand label (Left or Right)
                hand_label = handedness.classification[0].label
                
                # Get wrist position for text placement
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                text_x = int(wrist.x * w)
                text_y = int(wrist.y * h) - 20
                
                # Display hand side only
                draw_text_with_background(frame, f"{hand_label} Hand", 
                                         (text_x, text_y), font_scale=0.7,
                                         text_color=(0, 255, 255), bg_color=(50, 50, 50))
            
            # Display total hands detected
            num_hands = len(hand_results.multi_hand_landmarks)
            draw_text_with_background(frame, f"Hands: {num_hands}", 
                                     (w - 120, 30), font_scale=0.6,
                                     text_color=(0, 255, 255), bg_color=(50, 50, 50))
        
        # Process and display face detection
        if CONFIG["enable_face_detection"] and face_results and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw face mesh
                if CONFIG["show_face_mesh"]:
                    # Draw only contours for better performance
                    if CONFIG["face_mesh_tesselation"]:
                        # Full mesh (more detailed but slower)
                        mp_draw.draw_landmarks(
                            frame,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=face_connection_spec
                        )
                    else:
                        # Draw only face contours (faster)
                        mp_draw.draw_landmarks(
                            frame,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=face_drawing_spec,
                            connection_drawing_spec=face_connection_spec
                        )
                        
                        # Draw irises
                        mp_draw.draw_landmarks(
                            frame,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=face_drawing_spec,
                            connection_drawing_spec=face_connection_spec
                        )
                
                # Eye tracking and blink detection
                if CONFIG["enable_eye_tracking"]:
                    # Left eye landmarks (6 points for EAR calculation)
                    LEFT_EYE = [33, 160, 158, 133, 153, 144]
                    # Right eye landmarks
                    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
                    
                    # Get eye coordinates
                    left_eye_points = get_eye_landmarks(face_landmarks, LEFT_EYE, w, h)
                    right_eye_points = get_eye_landmarks(face_landmarks, RIGHT_EYE, w, h)
                    
                    # Calculate EAR for both eyes
                    left_ear = calculate_eye_aspect_ratio(left_eye_points)
                    right_ear = calculate_eye_aspect_ratio(right_eye_points)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Blink detection
                    if avg_ear < CONFIG["blink_threshold"]:
                        eye_closed_frames += 1
                        if not blink_detected and eye_closed_frames >= 2:
                            blink_detected = True
                            total_blinks += 1
                            last_blink_time = current_time
                    else:
                        if blink_detected:
                            blink_detected = False
                        eye_closed_frames = 0
                    
                    # Calculate blinks per minute
                    elapsed_minutes = (current_time - blink_start_time) / 60.0
                    if elapsed_minutes > 0:
                        blinks_per_minute = total_blinks / elapsed_minutes
                    
                    # Check for eye strain (too few blinks)
                    if elapsed_minutes >= 1 and blinks_per_minute < CONFIG["blink_alert_rate"]:
                        eye_strain_alert = True
                    else:
                        eye_strain_alert = False
                    
                    # Draw eye regions
                    for point in left_eye_points:
                        cv2.circle(frame, point, 2, (0, 255, 255), -1)
                    for point in right_eye_points:
                        cv2.circle(frame, point, 2, (0, 255, 255), -1)
                    
                    # Calculate and display gaze direction
                    if CONFIG["show_eye_gaze"]:
                        h_gaze, v_gaze = calculate_gaze_direction(face_landmarks, w, h)
                        
                        # Get nose tip for text placement
                        nose_tip = face_landmarks.landmark[1]
                        gaze_x = int(nose_tip.x * w)
                        gaze_y = int(nose_tip.y * h) + 30
                        
                        gaze_text = f"Gaze: {h_gaze}-{v_gaze}"
                        draw_text_with_background(frame, gaze_text, 
                                                 (gaze_x - 50, gaze_y), font_scale=0.6,
                                                 text_color=(255, 255, 0), bg_color=(50, 50, 0))
                    
                    # Display blink statistics
                    if CONFIG["track_blink_stats"]:
                        blink_y = 90
                        draw_text_with_background(frame, f"Blinks: {total_blinks}", 
                                                 (w - 150, blink_y), font_scale=0.6,
                                                 text_color=(0, 255, 255), bg_color=(50, 50, 50))
                        
                        draw_text_with_background(frame, f"BPM: {int(blinks_per_minute)}", 
                                                 (w - 150, blink_y + 25), font_scale=0.6,
                                                 text_color=(0, 255, 255), bg_color=(50, 50, 50))
                        
                        # Eye strain alert
                        if eye_strain_alert:
                            draw_text_with_background(frame, "Eye Strain Alert!", 
                                                     (w - 180, blink_y + 50), font_scale=0.6,
                                                     text_color=(0, 0, 255), bg_color=(100, 0, 0))
                            # Flash warning
                            cv2.rectangle(frame, (w - 200, blink_y - 10), 
                                        (w - 10, blink_y + 70), (0, 0, 255), 3)
                
                # Distance estimation
                if CONFIG["enable_distance_estimation"]:
                    face_width = calculate_face_width(face_landmarks, w)
                    estimated_distance = estimate_distance(
                        face_width, 
                        CONFIG["calibrated_face_width"], 
                        CONFIG["calibrated_distance"]
                    )
                    
                    if estimated_distance is not None:
                        # Determine distance status
                        if estimated_distance < CONFIG["min_safe_distance"]:
                            distance_status = "Too Close!"
                            distance_color = (0, 0, 255)
                            distance_bg = (100, 0, 0)
                            distance_alert = True
                        elif estimated_distance > CONFIG["max_safe_distance"]:
                            distance_status = "Too Far"
                            distance_color = (0, 165, 255)
                            distance_bg = (50, 50, 0)
                            distance_alert = True
                        else:
                            distance_status = "Good"
                            distance_color = (0, 255, 0)
                            distance_bg = (0, 100, 0)
                            distance_alert = False
                        
                        # Display distance
                        dist_y = 150
                        distance_text = f"Distance: {int(estimated_distance)} cm"
                        draw_text_with_background(frame, distance_text, 
                                                 (w - 200, dist_y), font_scale=0.7,
                                                 text_color=(255, 255, 255), bg_color=(50, 50, 50))
                        
                        status_text = f"Status: {distance_status}"
                        draw_text_with_background(frame, status_text, 
                                                 (w - 200, dist_y + 30), font_scale=0.7,
                                                 text_color=distance_color, bg_color=distance_bg)
                        
                        # Show warning if too close
                        if distance_alert and CONFIG["show_distance_warning"]:
                            if estimated_distance < CONFIG["min_safe_distance"]:
                                warning_text = "Move Back!"
                                cv2.rectangle(frame, (w - 220, dist_y - 10), 
                                            (w - 10, dist_y + 50), (0, 0, 255), 3)
                            else:
                                warning_text = "Move Closer"
                            
                            draw_text_with_background(frame, warning_text, 
                                                     (w - 200, dist_y + 60), font_scale=0.6,
                                                     text_color=(255, 255, 255), bg_color=(100, 0, 0))
                    else:
                        # Not calibrated
                        draw_text_with_background(frame, "Distance: Not Calibrated", 
                                                 (w - 220, 150), font_scale=0.6,
                                                 text_color=(128, 128, 128), bg_color=(50, 50, 50))
            
            # Display face detected indicator
            num_faces = len(face_results.multi_face_landmarks)
            draw_text_with_background(frame, f"Face: {num_faces}", 
                                     (w - 120, 60), font_scale=0.6,
                                     text_color=(0, 255, 0), bg_color=(0, 50, 0))
        
        # Display controls
        control_text = "Q:Quit | S:Screenshot | R:Reset | B:Posture | D:Distance | SPACE:Fullscreen"
        draw_text_with_background(frame, control_text, 
                                 (10, h - 10), font_scale=0.45,
                                 text_color=(200, 200, 200), bg_color=(0, 0, 0))
        
        # Logging
        if CONFIG["enable_logging"] and (current_time - last_log_time) >= CONFIG["log_interval"]:
            if result.pose_landmarks:
                add_log_entry("good" if CONFIG["good_posture_min_angle"] <= angle <= CONFIG["good_posture_max_angle"] else "bad", int(angle))
            last_log_time = current_time
        
        # Show the frame
        cv2.imshow(window_name, frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27 or key == ord('q'):  # ESC or Q to exit
            break
        elif key == 32:  # SPACE to toggle fullscreen
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Fullscreen enabled")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("Fullscreen disabled")
        elif key == ord('s'):  # Save screenshot
            filename = f"posture_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('r'):  # Reset statistics
            good_posture_time = 0
            bad_posture_time = 0
            bad_posture_continuous = 0
            posture_log = []
            print("Statistics reset!")
        elif key == ord('c'):  # Configuration menu
            print("\n=== Current Configuration ===")
            for key_name, value in CONFIG.items():
                print(f"{key_name}: {value}")
            print("============================\n")
        elif key == ord('b'):  # Recalibrate
            print("\nRecalibrating posture...")
            calibrate_posture(cap, pose)
            print("Calibration complete! Resuming detection...")
        elif key == ord('h'):  # Toggle hand detection
            CONFIG["enable_hand_detection"] = not CONFIG["enable_hand_detection"]
            status = "enabled" if CONFIG["enable_hand_detection"] else "disabled"
            print(f"Hand detection {status}")
        elif key == ord('f'):  # Toggle face detection
            CONFIG["enable_face_detection"] = not CONFIG["enable_face_detection"]
            status = "enabled" if CONFIG["enable_face_detection"] else "disabled"
            print(f"Face detection {status}")
        elif key == ord('m'):  # Toggle face mesh detail
            CONFIG["face_mesh_tesselation"] = not CONFIG["face_mesh_tesselation"]
            status = "full mesh" if CONFIG["face_mesh_tesselation"] else "contours only"
            print(f"Face mesh mode: {status}")
        elif key == ord('e'):  # Toggle eye tracking
            CONFIG["enable_eye_tracking"] = not CONFIG["enable_eye_tracking"]
            status = "enabled" if CONFIG["enable_eye_tracking"] else "disabled"
            print(f"Eye tracking {status}")
        elif key == ord('x'):  # Reset blink counter
            total_blinks = 0
            blink_start_time = current_time
            print("Blink counter reset!")
        elif key == ord('d'):  # Recalibrate distance
            print("\nRecalibrating distance...")
            calibrate_distance(cap, pose, face_mesh)
            print("Distance calibration complete! Resuming detection...")

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Cleanup and save
    save_log()
    
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()
    face_mesh.close()
    
    print("\n=== Session Summary ===")
    print(f"Good Posture Time: {int(good_posture_time)} seconds")
    print(f"Bad Posture Time: {int(bad_posture_time)} seconds")
    if good_posture_time + bad_posture_time > 0:
        percentage = (good_posture_time / (good_posture_time + bad_posture_time)) * 100
        print(f"Good Posture Percentage: {percentage:.1f}%")
    
    # Eye tracking summary
    if CONFIG["enable_eye_tracking"] and total_blinks > 0:
        print(f"\nEye Tracking Summary:")
        print(f"Total Blinks: {total_blinks}")
        print(f"Average Blinks Per Minute: {int(blinks_per_minute)}")
        print(f"Recommended: 15-20 blinks per minute")
    
    # Distance summary
    if CONFIG["enable_distance_estimation"] and estimated_distance > 0:
        print(f"\nDistance Summary:")
        print(f"Last Measured Distance: {int(estimated_distance)} cm")
        print(f"Recommended Range: {CONFIG['min_safe_distance']}-{CONFIG['max_safe_distance']} cm (8-16 inches)")
    
    print("======================")
