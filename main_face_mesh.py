import cv2
import mediapipe as mp
import math
import time
import json
import numpy as np
import random
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
    "enable_face_detection": True,   # Enable for side box display
    "max_faces": 1,
    "face_detection_confidence": 0.5,
    "face_tracking_confidence": 0.5,
    "show_face_mesh": False,         # Don't show on actual face
    "show_face_mesh_in_box": True,   # Show in side box
    "face_mesh_tesselation": True,   # Full detailed mesh
    
    # Eye Tracking Settings
    "enable_eye_tracking": False,    # Disable eye tracking text
    "show_eye_gaze": False,          # Don't show gaze text
    "blink_threshold": 0.21,
    "blink_alert_rate": 15,
    "track_blink_stats": False,      # Don't show blink stats
    
    # Distance Estimation Settings
    "enable_distance_estimation": False,  # Disable distance text
    "calibrated_distance": None,
    "calibrated_face_width": None,
    "min_safe_distance": 20,
    "max_safe_distance": 40,
    "show_distance_warning": False,  # Don't show distance warnings
    
    # Alert Settings
    "bad_posture_alert_delay": 10,  # Seconds of bad posture before alert
    "enable_visual_alert": True,    # Flash screen on bad posture
    "enable_sound_alert": False,    # Beep sound (requires additional setup)
    
    # Display Settings
    "show_skeleton": True,          # Show pose landmarks
    "show_angles": False,           # Don't show angle measurements
    "show_statistics": False,       # Don't show time statistics
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

# Initialize cosmic background
def init_cosmic_background(width, height):
    """Initialize cosmic cloud and star particles"""
    global cloud_particles, star_particles
    
    # Create cloud particles
    cloud_particles = []
    for _ in range(150):  # Number of cloud particles
        particle = {
            'x': random.uniform(0, width),
            'y': random.uniform(0, height),
            'size': random.uniform(20, 80),
            'speed': random.uniform(0.2, 0.8),
            'color_intensity': random.uniform(0.3, 1.0),
            'drift_x': random.uniform(-0.5, 0.5),
            'drift_y': random.uniform(-0.5, 0.5),
            'phase': random.uniform(0, 2 * math.pi)
        }
        cloud_particles.append(particle)
    
    # Create star particles
    star_particles = []
    for _ in range(200):  # Number of stars
        star = {
            'x': random.uniform(0, width),
            'y': random.uniform(0, height),
            'brightness': random.uniform(0.3, 1.0),
            'twinkle_speed': random.uniform(2, 6),
            'size': random.randint(1, 3),
            'phase': random.uniform(0, 2 * math.pi)
        }
        star_particles.append(star)

def draw_cosmic_background(frame):
    """Draw professional dark background for point cloud visualization"""
    global cosmic_time
    
    h, w = frame.shape[:2]
    cosmic_time += 0.02
    
    # Create professional dark background with subtle grid
    dark_overlay = np.zeros_like(frame)
    
    # Add subtle grid pattern for professional look
    grid_spacing = 50
    grid_alpha = 30
    
    # Vertical grid lines
    for x in range(0, w, grid_spacing):
        cv2.line(dark_overlay, (x, 0), (x, h), (grid_alpha, grid_alpha, grid_alpha), 1)
    
    # Horizontal grid lines
    for y in range(0, h, grid_spacing):
        cv2.line(dark_overlay, (0, y), (w, y), (grid_alpha, grid_alpha, grid_alpha), 1)
    
    # Blend with camera feed for professional dark background
    frame[:] = cv2.addWeighted(frame, 0.2, dark_overlay, 0.8, 0)

# Initialize webcam
cap = cv2.VideoCapture(CONFIG["camera_id"])

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])

# Initialize cosmic background
init_cosmic_background(CONFIG["frame_width"], CONFIG["frame_height"])

# Variables for tracking (simplified - no posture tracking)
last_time = time.time()
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

# Cosmic cloud background variables
cloud_particles = []
star_particles = []
nebula_offset = 0
cosmic_time = 0

# Face mesh control configuration
face_mesh_config = {
    "show_face_mesh": True,
    "mesh_density": 1.0,
    "point_size": 2,
    "show_contours": False,
    "show_key_points": False
}

# Gesture control variables
current_gesture = "UNKNOWN"
gesture_status = "Ready"
gesture_confidence = 0.0

def draw_face_point_cloud(frame, face_landmarks, w, h):
    """Draw dense green point cloud visualization of face"""
    
    # Face point cloud position (left side of screen) - reduced size
    cloud_center_x = w // 4
    cloud_center_y = h // 2
    cloud_scale = 2.2  # Reduced from 3.0 to 2.2 for smaller size
    
    # Get face center for positioning
    nose = face_landmarks.landmark[1]
    face_center_x = nose.x * w
    face_center_y = nose.y * h
    
    # Draw all face landmarks as green point cloud
    for i, landmark in enumerate(face_landmarks.landmark):
        # Calculate 3D position with depth
        depth_factor = 1 - (landmark.z * 2)  # Convert z to depth factor
        depth_factor = max(0.3, min(1.0, depth_factor))
        
        # Position in point cloud area
        point_x = cloud_center_x + (landmark.x * w - face_center_x) * cloud_scale
        point_y = cloud_center_y + (landmark.y * h - face_center_y) * cloud_scale
        
        # Point size based on depth and importance
        if i in [1, 33, 263, 61, 291, 10, 151]:  # Key landmarks
            point_size = int(4 * depth_factor)
            brightness = int(255 * depth_factor)
        else:
            point_size = int(2 * depth_factor)
            brightness = int(200 * depth_factor)
        
        # Matrix-style green color with depth variation
        green_intensity = max(100, brightness)
        
        # Draw point with glow effect
        cv2.circle(frame, (int(point_x), int(point_y)), point_size + 1, 
                  (0, green_intensity//3, 0), -1)  # Dark green glow
        cv2.circle(frame, (int(point_x), int(point_y)), point_size, 
                  (0, green_intensity, 0), -1)  # Bright green core
    
    # Add interpolated points for denser cloud
    for connection in mp_face_mesh.FACEMESH_CONTOURS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start = face_landmarks.landmark[start_idx]
        end = face_landmarks.landmark[end_idx]
        
        # Interpolate points between landmarks
        for t in np.linspace(0, 1, 5):  # 5 points between each connection
            interp_x = start.x + t * (end.x - start.x)
            interp_y = start.y + t * (end.y - start.y)
            interp_z = start.z + t * (end.z - start.z)
            
            depth_factor = 1 - (interp_z * 2)
            depth_factor = max(0.3, min(1.0, depth_factor))
            
            point_x = cloud_center_x + (interp_x * w - face_center_x) * cloud_scale
            point_y = cloud_center_y + (interp_y * h - face_center_y) * cloud_scale
            
            point_size = max(1, int(2 * depth_factor))
            brightness = int(150 * depth_factor)
            
            cv2.circle(frame, (int(point_x), int(point_y)), point_size, 
                      (0, brightness, 0), -1)

def analyze_hand_gesture(hand_landmarks):
    """Analyze hand gesture and return gesture type and confidence"""
    
    # Get landmark positions
    landmarks = hand_landmarks.landmark
    
    # Helper function to check if finger is extended (improved logic)
    def is_finger_extended(tip_idx, pip_idx, mcp_idx):
        tip_y = landmarks[tip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        # More lenient threshold for finger extension
        return tip_y < (pip_y - 0.02)  # Added threshold for better detection
    
    # Helper function to check if thumb is extended (improved)
    def is_thumb_extended():
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        # Check both x and y coordinates for better thumb detection
        return (thumb_tip.x > thumb_ip.x) and (thumb_tip.y < thumb_mcp.y)
    
    # Count extended fingers
    fingers_up = []
    
    # Thumb (more reliable detection)
    fingers_up.append(1 if is_thumb_extended() else 0)
    
    # Other fingers (Index, Middle, Ring, Pinky)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    finger_mcps = [5, 9, 13, 17]
    
    for i in range(4):
        fingers_up.append(1 if is_finger_extended(finger_tips[i], finger_pips[i], finger_mcps[i]) else 0)
    
    total_fingers = sum(fingers_up)
    
    # More lenient gesture recognition with debug info
    if total_fingers == 0:
        return "FIST", 0.8, fingers_up
    elif total_fingers == 1 and fingers_up[1] == 1:  # Only index finger
        return "POINT", 0.9, fingers_up
    elif total_fingers == 2 and fingers_up[1] == 1 and fingers_up[2] == 1:  # Index + Middle
        return "PEACE", 0.9, fingers_up
    elif total_fingers == 3:
        return "THREE", 0.8, fingers_up
    elif total_fingers == 4:
        return "FOUR", 0.8, fingers_up
    elif total_fingers == 5:
        return "OPEN_HAND", 0.9, fingers_up
    else:
        # Default to showing face mesh for unknown gestures
        return "SHOW_DEFAULT", 0.7, fingers_up

def apply_gesture_control(gesture, face_mesh_config):
    """Apply gesture-based controls to face mesh visualization"""
    
    if gesture == "FIST":
        # Fist: Minimal face mesh (not hidden completely)
        face_mesh_config["show_face_mesh"] = True
        face_mesh_config["mesh_density"] = 0.2
        face_mesh_config["point_size"] = 1
        return "Face Mesh: MINIMAL"
    
    elif gesture == "POINT":
        # Point: Low density face mesh
        face_mesh_config["show_face_mesh"] = True
        face_mesh_config["mesh_density"] = 0.4
        face_mesh_config["point_size"] = 2
        return "Face Mesh: LOW"
    
    elif gesture == "PEACE":
        # Peace: Medium density face mesh
        face_mesh_config["show_face_mesh"] = True
        face_mesh_config["mesh_density"] = 0.6
        face_mesh_config["point_size"] = 2
        return "Face Mesh: MEDIUM"
    
    elif gesture == "THREE":
        # Three: High density face mesh
        face_mesh_config["show_face_mesh"] = True
        face_mesh_config["mesh_density"] = 0.8
        face_mesh_config["point_size"] = 3
        return "Face Mesh: HIGH"
    
    elif gesture == "FOUR":
        # Four: Maximum density face mesh
        face_mesh_config["show_face_mesh"] = True
        face_mesh_config["mesh_density"] = 1.0
        face_mesh_config["point_size"] = 3
        return "Face Mesh: MAXIMUM"
    
    elif gesture == "OPEN_HAND":
        # Open hand: Full face mesh with all features
        face_mesh_config["show_face_mesh"] = True
        face_mesh_config["mesh_density"] = 1.0
        face_mesh_config["point_size"] = 3
        face_mesh_config["show_contours"] = True
        face_mesh_config["show_key_points"] = True
        return "Face Mesh: FULL FEATURES"
    
    else:
        # Default: Always show face mesh
        face_mesh_config["show_face_mesh"] = True
        face_mesh_config["mesh_density"] = 0.8
        face_mesh_config["point_size"] = 2
        return "Face Mesh: DEFAULT"

def draw_face_point_cloud_controlled(frame, face_landmarks, w, h, mesh_config):
    """Draw face point cloud with gesture-controlled parameters"""
    
    if not mesh_config.get("show_face_mesh", True):
        return
    
    # Face point cloud position (left side of screen)
    cloud_center_x = w // 4
    cloud_center_y = h // 2
    cloud_scale = 2.2
    
    # Get face center for positioning
    nose = face_landmarks.landmark[1]
    face_center_x = nose.x * w
    face_center_y = nose.y * h
    
    # Get mesh parameters from config
    density = mesh_config.get("mesh_density", 1.0)
    point_size = mesh_config.get("point_size", 2)
    
    # Draw face landmarks based on density
    landmark_step = max(1, int(1 / density)) if density > 0 else len(face_landmarks.landmark)
    
    for i in range(0, len(face_landmarks.landmark), landmark_step):
        landmark = face_landmarks.landmark[i]
        
        # Calculate 3D position with depth
        depth_factor = 1 - (landmark.z * 2)
        depth_factor = max(0.3, min(1.0, depth_factor))
        
        # Position in point cloud area
        point_x = cloud_center_x + (landmark.x * w - face_center_x) * cloud_scale
        point_y = cloud_center_y + (landmark.y * h - face_center_y) * cloud_scale
        
        # Adjust point size based on config and depth
        current_size = int(point_size * depth_factor)
        brightness = int(255 * depth_factor * density)
        
        # Matrix-style green color with controlled brightness
        green_intensity = max(50, brightness)
        
        # Draw point with glow effect
        if current_size > 0:
            cv2.circle(frame, (int(point_x), int(point_y)), current_size + 1, 
                      (0, green_intensity//3, 0), -1)  # Dark green glow
            cv2.circle(frame, (int(point_x), int(point_y)), current_size, 
                      (0, green_intensity, 0), -1)  # Bright green core
    
    # Draw contours if enabled
    if mesh_config.get("show_contours", False) and density > 0.5:
        for connection in mp_face_mesh.FACEMESH_CONTOURS:
            if np.random.random() < density:  # Random sampling based on density
                start_idx = connection[0]
                end_idx = connection[1]
                
                start = face_landmarks.landmark[start_idx]
                end = face_landmarks.landmark[end_idx]
                
                start_x = cloud_center_x + (start.x * w - face_center_x) * cloud_scale
                start_y = cloud_center_y + (start.y * h - face_center_y) * cloud_scale
                end_x = cloud_center_x + (end.x * w - face_center_x) * cloud_scale
                end_y = cloud_center_y + (end.y * h - face_center_y) * cloud_scale
                
                cv2.line(frame, (int(start_x), int(start_y)), (int(end_x), int(end_y)), 
                        (0, int(150 * density), 0), 1)
    
    # Draw key points if enabled
    if mesh_config.get("show_key_points", False):
        key_landmarks = [1, 33, 263, 61, 291, 10, 151]  # Key facial points
        
        for landmark_idx in key_landmarks:
            if landmark_idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[landmark_idx]
                
                point_x = cloud_center_x + (landmark.x * w - face_center_x) * cloud_scale
                point_y = cloud_center_y + (landmark.y * h - face_center_y) * cloud_scale
                
                # Bright key points
                cv2.circle(frame, (int(point_x), int(point_y)), point_size + 2, 
                          (0, 255, 0), -1)
                cv2.circle(frame, (int(point_x), int(point_y)), point_size + 1, 
                          (100, 255, 100), -1)

def draw_gesture_status(frame, gesture, status_text, w, h, fingers_debug=None):
    """Draw gesture recognition status - DISABLED for clean screen"""
    # All text displays removed for clean interface
    pass
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

def add_log_entry(status, info):
    """Add entry to log (simplified)"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "info": info
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
print("Advanced Face and Hand Tracking Started!")
print("\nFace and Hand Biometric Analysis System")
print("Features:")
print("  - Dense green point cloud face visualization")
print("  - Detailed hand skeleton tracking")
print("  - Professional dark background")
print("  - Real-time 3D positioning")

print("\n=== Controls ===")
print("Q or ESC: Exit")
print("S: Save screenshot")
print("R: Reset statistics")
print("H: Toggle hand detection")
print("F: Toggle face detection")
print("SPACE: Toggle fullscreen")
print("================\n")

# Create window and set to fullscreen
window_name = "Advanced Face and Hand Tracking"
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
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Convert BGR to RGB for MediaPipe processing
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
        
        # Draw cosmic background overlay after processing
        draw_cosmic_background(frame)
        
        # Track time
        current_time = time.time()
        elapsed = current_time - last_time
        last_time = current_time
        
        if result.pose_landmarks:
            # Posture detection removed - keeping only face and hand tracking
            pass
        # No text displays
        
        # Process and display hand detection with gesture control
        if CONFIG["enable_hand_detection"] and hand_results and hand_results.multi_hand_landmarks:
            for hand_idx, (hand_landmarks, handedness) in enumerate(zip(
                hand_results.multi_hand_landmarks, 
                hand_results.multi_handedness
            )):
                # Analyze hand gesture with debug info
                gesture, confidence, fingers_debug = analyze_hand_gesture(hand_landmarks)
                
                # Apply gesture control to face mesh (with lower confidence threshold)
                if confidence > 0.5:  # Lowered threshold for better responsiveness
                    current_gesture = gesture
                    gesture_confidence = confidence
                    gesture_status = apply_gesture_control(gesture, face_mesh_config)
                
                # Draw gesture status - DISABLED for clean screen
                # draw_gesture_status(frame, current_gesture, gesture_status, w, h, fingers_debug)
                
                # Get wrist position
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                text_x = int(wrist.x * w)
                text_y = int(wrist.y * h) - 20
                
                # No hand text displays - clean screen
        
        # Process and display face detection with gesture-controlled point cloud
        if CONFIG["enable_face_detection"] and face_results and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw face point cloud with gesture controls
                draw_face_point_cloud_controlled(frame, face_landmarks, w, h, face_mesh_config)
        
        # No control text display - clean screen
        pass
        
        # Logging (simplified)
        if CONFIG["enable_logging"] and (current_time - last_log_time) >= CONFIG["log_interval"]:
            add_log_entry("tracking", "face_and_hand_active")
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
            posture_log = []
            print("Statistics reset!")
        elif key == ord('h'):  # Toggle hand detection
            CONFIG["enable_hand_detection"] = not CONFIG["enable_hand_detection"]
            status = "enabled" if CONFIG["enable_hand_detection"] else "disabled"
            print(f"Hand detection {status}")
        elif key == ord('f'):  # Toggle face detection
            CONFIG["enable_face_detection"] = not CONFIG["enable_face_detection"]
            status = "enabled" if CONFIG["enable_face_detection"] else "disabled"
            print(f"Face detection {status}")

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
    print("Face and Hand Tracking Session Complete")
    if posture_log:
        print(f"Total tracking entries: {len(posture_log)}")
    print("======================")
