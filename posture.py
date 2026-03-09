import cv2
import mediapipe as mp
import math
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
# Use static_image_mode=False for video, increase detection confidence
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    print("Please check if your webcam is connected and not being used by another application")
    exit()

print("Webcam connected successfully!")

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Variables for tracking
posture_data = {}

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return abs(angle)

def detect_posture_from_side(landmarks, w, h):
    """Detect posture from side/angled view"""
    try:
        # Get key landmarks
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Use average of left and right for better accuracy
        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2 * w
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * h
        hip_x = (left_hip.x + right_hip.x) / 2 * w
        hip_y = (left_hip.y + right_hip.y) / 2 * h
        nose_x = nose.x * w
        nose_y = nose.y * h
        
        # Calculate vertical alignment
        # Check if head is forward relative to shoulders
        head_forward = abs(nose_y - shoulder_y)
        
        # Check if shoulders are forward relative to hips
        shoulder_forward = abs(shoulder_y - hip_y)
        
        # Calculate the angle
        angle = calculate_angle(
            [nose_x, nose_y],
            [shoulder_x, shoulder_y],
            [hip_x, hip_y]
        )
        
        # Determine posture based on angle and position
        # For side view: slouching forward shows smaller angle
        is_good_posture = 150 <= angle <= 175
        
        return {
            'angle': angle,
            'is_good': is_good_posture,
            'shoulder': (int(shoulder_x), int(shoulder_y)),
            'hip': (int(hip_x), int(hip_y)),
            'nose': (int(nose_x), int(nose_y))
        }
    except:
        return None

print("Posture Detection Started!")
print("Press ESC to exit")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    frame_count += 1
    
    # Process every frame for better detection
    h, w, _ = frame.shape
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    result = pose.process(rgb)
    
    if result.pose_landmarks:
        # Draw pose landmarks
        mp_draw.draw_landmarks(
            frame, 
            result.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        
        # Detect posture
        posture_info = detect_posture_from_side(result.pose_landmarks.landmark, w, h)
        
        if posture_info:
            # Draw key points
            cv2.circle(frame, posture_info['nose'], 8, (255, 0, 0), -1)
            cv2.circle(frame, posture_info['shoulder'], 8, (255, 0, 0), -1)
            cv2.circle(frame, posture_info['hip'], 8, (255, 0, 0), -1)
            
            # Display angle
            cv2.putText(frame, f"Angle: {int(posture_info['angle'])}", 
                       (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Determine posture status
            if posture_info['is_good']:
                cv2.putText(frame, "Good Posture", 
                           (posture_info['shoulder'][0] - 50, posture_info['shoulder'][1] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, 
                            (posture_info['nose'][0] - 100, posture_info['nose'][1] - 100),
                            (posture_info['hip'][0] + 100, posture_info['hip'][1] + 100),
                            (0, 255, 0), 3)
            else:
                cv2.putText(frame, "Bad Posture!", 
                           (posture_info['shoulder'][0] - 50, posture_info['shoulder'][1] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(frame, 
                            (posture_info['nose'][0] - 100, posture_info['nose'][1] - 100),
                            (posture_info['hip'][0] + 100, posture_info['hip'][1] + 100),
                            (0, 0, 255), 3)
    else:
        # No pose detected
        cv2.putText(frame, "No person detected clearly", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Display instructions
    cv2.putText(frame, "Press ESC to exit", (w - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show the frame
    cv2.imshow("Posture Detection - RTSP Stream", frame)
    
    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()

print("\nPosture detection ended")