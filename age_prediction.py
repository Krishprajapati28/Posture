import cv2
import numpy as np

class AgePredictionModel:
    def __init__(self):
        """Initialize the age prediction model with pre-trained weights"""
        # Original age ranges from the model
        self.model_age_ranges = ['(0-2)', '(4-6)', '(8-12)','(12-18)','(18-23)', 
                                 '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        # Refined age ranges for display (closer ranges)
        self.age_ranges = ['0-2', '4-6', '8-12', '(12-18)','(18-23)', 
                          '25-32', '38-43', '48-53', '60+']
        
        # Load pre-trained models
        self.face_net = None
        self.age_net = None
        self.load_models()
    
    def load_models(self):
        """Load face detection and age prediction models"""
        try:
            # Face detection model
            face_proto = "opencv_face_detector.pbtxt"
            face_model = "opencv_face_detector_uint8.pb"
            self.face_net = cv2.dnn.readNet(face_model, face_proto)
            
            # Age prediction model
            age_proto = "age_deploy.prototxt"
            age_model = "age_net.caffemodel"
            self.age_net = cv2.dnn.readNet(age_model, age_proto)
            
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please download the required model files (see README)")
    
    def detect_faces(self, image):
        """Detect faces in the image"""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), 
                                     (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                faces.append(box.astype(int))
        
        return faces
    
    def predict_age(self, image, face_box):
        """Predict age for a detected face"""
        x1, y1, x2, y2 = face_box
        
        # Add padding to face region for better accuracy
        padding = 20
        y1 = max(0, y1 - padding)
        y2 = min(image.shape[0], y2 + padding)
        x1 = max(0, x1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return "Unknown"
        
        # Preprocess face for age prediction
        blob = cv2.dnn.blobFromImage(
            face, 
            scalefactor=1.0, 
            size=(227, 227),
            mean=(78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False,
            crop=False
        )
        
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        
        # Get top 2 predictions for better accuracy
        top_indices = age_preds[0].argsort()[-2:][::-1]
        top_idx = top_indices[0]
        confidence = age_preds[0][top_idx]
        
        # If confidence is low, show range between top 2 predictions
        if confidence < 0.5 and len(top_indices) > 1:
            second_idx = top_indices[1]
            age_range = f"{self.age_ranges[min(top_idx, second_idx)]}-{self.age_ranges[max(top_idx, second_idx)]}"
            combined_confidence = (age_preds[0][top_idx] + age_preds[0][second_idx]) / 2
            return age_range, combined_confidence
        
        # Return single age range with confidence
        return self.age_ranges[top_idx], confidence
    
    def process_image(self, image_path):
        """Process an image and predict ages"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            print("No faces detected in the image")
            return
        
        results = []
        for i, face in enumerate(faces):
            age, confidence = self.predict_age(image, face)
            results.append({
                'face_id': i + 1,
                'age_range': age,
                'confidence': confidence,
                'bbox': face
            })
            
            # Draw on image
            x1, y1, x2, y2 = face
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display age with confidence
            label = f"Age: {age} ({confidence*100:.1f}%)"
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save result
        output_path = "output_" + image_path.split('/')[-1]
        cv2.imwrite(output_path, image)
        print(f"Result saved to {output_path}")
        
        return results
    
    def process_webcam(self):
        """Real-time age prediction from webcam"""
        cap = cv2.VideoCapture(0)
        
        print("Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = self.detect_faces(frame)
            
            for face in faces:
                age, confidence = self.predict_age(frame, face)
                x1, y1, x2, y2 = face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"Age: {age} ({confidence*100:.1f}%)"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Age Prediction', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    model = AgePredictionModel()
    
    if len(sys.argv) > 1:
        # Process image file
        image_path = sys.argv[1]
        results = model.process_image(image_path)
        if results:
            print("\nDetection Results:")
            for result in results:
                print(f"Face {result['face_id']}: Age {result['age_range']} "
                      f"(Confidence: {result['confidence']*100:.1f}%)")
    else:
        # Use webcam
        print("No image provided. Starting webcam mode...")
        model.process_webcam()
