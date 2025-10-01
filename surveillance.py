# surveillance.py

import os
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import datetime

# ==============================================================================
# 1. DEEP LEARNING MODEL DEFINITION (MUST MATCH THE TRAINING SCRIPT)
# ==============================================================================
# This class defines the same architecture as the model we trained.
# It is necessary to create an instance of this class before we can load the saved weights.
class PersonDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(PersonDetector, self).__init__()
        # We load the ResNet18 architecture, but not its pre-trained weights (weights=None).
        # The weights will be loaded from our trained .pth file.
        self.backbone = torchvision.models.resnet18(weights=None)

        num_ftrs = self.backbone.fc.in_features
        # The custom classification head must be identical to the one in the training script.
        self.backbone.fc = nn.Sequential( # type: ignore
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ==============================================================================
# 2. SURVEILLANCE SYSTEM COMPONENTS
# ==============================================================================
class MotionDetector:
    """Detects motion in a video stream using adaptive background subtraction."""
    def __init__(self, threshold=25, min_area=500):
        self.background = None
        self.threshold = threshold
        self.min_area = min_area

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.background is None:
            self.background = gray.copy().astype("float")
            return [], False

        # Update background model using a running average
        cv2.accumulateWeighted(gray, self.background, 0.5)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))

        # Threshold the delta image, then perform dilation to fill in holes
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2) # type: ignore

        # Find contours on thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        motion_areas = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        return motion_areas, len(motion_areas) > 0

class PersonClassifier:
    """Classifies regions of interest (ROIs) using the trained PersonDetector model."""
    def __init__(self, model_path, device):
        self.device = device
        self.model = PersonDetector(num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Set model to evaluation mode

        # This transform must be identical to the validation transform in the training script
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify_region(self, frame, bbox):
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]

        if roi.size == 0:
            return False, 0.0

        try:
            # Prepare image for the model
            input_tensor = self.transform(roi).unsqueeze(0).to(self.device)

            with torch.no_grad(): # Disable gradient calculation for faster inference
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                # Get the confidence of the 'person' class (index 1)
                confidence = probabilities[0][1].item()

            # Use a confidence threshold to decide if it's a person
            # This is set higher to reduce false alarms, especially since the model overfit slightly.
            is_person = confidence > 0.85
            return is_person, confidence
        except Exception:
            return False, 0.0

# ==============================================================================
# 3. MAIN SURVEILLANCE SYSTEM
# ==============================================================================
class SurveillanceSystem:
    """Integrates all components to run the live intruder detection system."""
    def __init__(self, model_path, device, camera_id=0):
        self.motion_detector = MotionDetector(threshold=30, min_area=1000)
        self.person_classifier = PersonClassifier(model_path, device)
        self.camera_id = camera_id

    def run_surveillance(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("Error: Could not open camera. Please check if your webcam is connected.")
            return

        print("\nSurveillance system started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            motion_areas, _ = self.motion_detector.detect_motion(frame)
            person_detections = []

            for contour in motion_areas:
                bbox = cv2.boundingRect(contour)
                is_person, confidence = self.person_classifier.classify_region(frame, bbox)

                if is_person:
                    x, y, w, h = bbox
                    person_detections.append((x, y, w, h, confidence))
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"ALERT: Intruder detected at {timestamp} with confidence {confidence:.2f}")

            display_frame = self.draw_overlays(frame, motion_areas, person_detections)
            cv2.imshow('Surveillance System', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\nSurveillance stopped.")

    def draw_overlays(self, frame, motion_areas, detections):
        # Draw green boxes for all detected motion
        for contour in motion_areas:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw red boxes and alerts for confirmed person detections
        is_intruder_detected = False
        for (x, y, w, h, confidence) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            label = f"PERSON: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            is_intruder_detected = True

        if is_intruder_detected:
            cv2.putText(frame, "INTRUDER ALERT!", (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2)

        return frame

# ==============================================================================
# 4. SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    MODEL_PATH = 'person_detector_model.pth'

    # Automatically select CPU, as most local machines for testing won't have a configured GPU
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Critical check to ensure the model file exists before starting
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please ensure the trained model is in the same directory as this script.")
    else:
        surveillance_system = SurveillanceSystem(model_path=MODEL_PATH, device=device)
        surveillance_system.run_surveillance()