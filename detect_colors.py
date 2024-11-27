import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load your trained YOLOv8 model
model = YOLO('runs/detect/train/weights/best.pt')

# Create a directory to save marked frames
os.makedirs("marked_frames", exist_ok=True)

# Function to calculate the average color of an object
def get_average_color(image):
    """Find the average color of an image."""
    average_color = np.mean(image, axis=(0, 1))  # Calculate mean color of the image (BGR)
    return tuple(map(int, average_color))  # Return as tuple of integers

# Run predictions on frames
results = model.predict(source='frames/', save=False, conf=0.5)

# Process each frame
for idx, result in enumerate(results):
    frame = result.orig_img  # Original frame image
    detections = result.boxes  # Detected objects

    if len(detections) == 0:
        # Skip frames with no detections
        continue

    for box in detections.xyxy:  # xyxy bounding box format
        x1, y1, x2, y2 = map(int, box[:4])

        # Crop the object region
        cropped_object = frame[y1:y2, x1:x2]

        # Get the average color of the cropped object
        average_color = get_average_color(cropped_object)

        # Draw the bounding box with the average color of the object
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=average_color, thickness=2)

        # Use the same color for the label
        color_label = f"Color: {average_color}"
        cv2.putText(frame, color_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, average_color, 2)

    # Save the marked frame
    output_path = f"marked_frames/frame_{idx}.jpg"
    cv2.imwrite(output_path, frame)

print("Marked frames saved in the 'marked_frames/' directory.")
