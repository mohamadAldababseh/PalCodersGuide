from image_classifier import ImageClassifier
import cv2
import threading
import time

# ==========================
# DEFINE CROP COORDINATES
# ==========================
crop_x1 = 300
crop_y1 = 100
crop_x2 = 530
crop_y2 = 280

# ==========================
# SET YOUR CLASSIFIER
# ==========================



# 🟢 Cropped-area classification  
#Camera default id is 0 
classifier = ImageClassifier.from_crop(
    model_path="model.tflite",
    labels_path="labels.txt",
    x1=crop_x1,
    y1=crop_y1,
    x2=crop_x2,
    y2=crop_y2
)


#you can pass cam id like this ...
'''
# Pass the camera_id to the classifier so it uses the selected camera
classifier = ImageClassifier.from_crop(
    model_path="model.tflite",
    labels_path="labels.txt",
    x1=crop_x1,
    y1=crop_y1,
    x2=crop_x2,
    y2=crop_y2,
    camera_id=camera_id  # <-- your chosen camera
)

'''

# ==========================
# Start classifier thread
# ==========================
threading.Thread(target=classifier.run, daemon=True).start()

# ==========================
# FPS variables
# ==========================
prev_time = time.time()
fps = 0.0

# ==========================
# Main loop
# ==========================
while True:
    frame, label, conf = classifier.get_data()

    # Wait until first frame is ready
    if frame is None:
        time.sleep(0.01)
        continue

    # ------------------- FPS -------------------
    now = time.time()
    dt = now - prev_time
    prev_time = now
    fps = fps * 0.9 + (1.0 / dt) * 0.1 if dt > 0 else fps

    # ------------------- Confidence 0-100 -------------------
    conf_percent = conf * 100 + 0.5
    conf_percent = max(0, min(100, conf_percent))

    # ------------------- Draw ROI (for cropped classifier) -------------------
        # Check if the classifier object has the attribute 'use_crop'
    # and if it is set to True. 
    # - hasattr() prevents an AttributeError in case the attribute doesn't exist.
    # - use_crop is True only if we created the classifier with the cropped-area constructor.
    # This ensures the following code for drawing the crop rectangle
    # only runs when the classifier is actually using a cropped input.


    if hasattr(classifier, 'use_crop') and classifier.use_crop:
        x1, y1, x2, y2 = classifier.crop_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        # Display the cropped area in a separate window
        cropped = frame[y1:y2, x1:x2]
        cv2.imshow("Cropped Area", cropped)

    # ------------------- Draw label + confidence -------------------
    cv2.putText(frame, f"{label}: {conf_percent}%", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ------------------- Draw FPS -------------------
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # ------------------- Show main camera -------------------
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.03)

# ==========================
# Cleanup
# ==========================
classifier.stop()
classifier.release()
cv2.destroyAllWindows()
