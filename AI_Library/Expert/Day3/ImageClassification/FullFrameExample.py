from image_classifier import ImageClassifier
import cv2
import threading
import time

# ==========================
# SET YOUR CLASSIFIER
# ==========================

#================== 🔵 Full-frame classification  
#default cam id is 0 
classifier = ImageClassifier.from_full_frame(model_path="model.tflite",labels_path="labels.txt")
#you can pass cam id lik this ...
#classifier = ImageClassifier.from_full_frame(
#    model_path="model.tflite",
#    labels_path="labels.txt",
#    camera_id=1  # use camera 1 instead of default 0
#)
# use camera 1 instead of default 0)


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
