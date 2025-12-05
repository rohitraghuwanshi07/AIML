import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n')  # Ultralytics downloads weights automatically
print("[INFO] YOLOv8 model loaded ✅")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("[ERROR] Could not open webcam.")

print("[INFO] Webcam initialized. Press 'q' to quit.")

# Live detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Run detection + tracking
    results = model.track(frame, persist=True)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Show live feed
    cv2.imshow("YOLOv8 Live Tracking", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam released ✅")


