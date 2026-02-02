import cv2
import numpy as np
import tensorflow as tf
import threading
import time


class ImageClassifier:
    # -------------------------------------------------
    # MAIN CONSTRUCTOR (PRIVATE USE)
    # -------------------------------------------------
    def __init__(self,
                 model_path,
                 labels_path,
                 use_crop=False,
                 crop_rect=None):

        self.use_crop = use_crop
        self.crop_rect = crop_rect  # (x1, y1, x2, y2)

        # ---------------- Camera ----------------
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc(*"MJPG"))

        if not self.cap.isOpened():
            raise RuntimeError("Camera not found")

        # ---------------- Labels ----------------
        with open(labels_path, "r") as f:
            self.class_names = [l.strip() for l in f.readlines()]

        # ---------------- Model ----------------
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_dtype = self.input_details[0]["dtype"]
        self.img_size = self.input_details[0]["shape"][1]

        # ---------------- Shared data ----------------
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_label = "None"
        self.latest_conf = 0.0
        self.running = False

    # =================================================
    # 🔵 CONSTRUCTOR 1: FULL FRAME CLASSIFIER
    # =================================================
    @classmethod
    def from_full_frame(cls, model_path, labels_path):
        return cls(
            model_path=model_path,
            labels_path=labels_path,
            use_crop=False
        )

    # =================================================
    # 🟢 CONSTRUCTOR 2: CROPPED AREA CLASSIFIER
    # =================================================
    @classmethod
    def from_crop(cls,
                  model_path,
                  labels_path,
                  x1, y1, x2, y2):
        return cls(
            model_path=model_path,
            labels_path=labels_path,
            use_crop=True,
            crop_rect=(x1, y1, x2, y2)
        )

    # ---------------- Preprocess ----------------
    def preprocess(self, img):
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(self.input_dtype)
        img = np.expand_dims(img, axis=0)
        return img

    # ---------------- One inference ----------------
    def classify_once(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Choose input area
        if self.use_crop and self.crop_rect is not None:
            x1, y1, x2, y2 = self.crop_rect
            input_img = frame[y1:y2, x1:x2]
        else:
            input_img = frame

        input_tensor = self.preprocess(input_img)
        self.interpreter.set_tensor(
            self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        raw = self.interpreter.get_tensor(
            self.output_details[0]["index"])[0]

        # -------- Softmax --------
        exp = np.exp(raw - np.max(raw))
        probs = exp / np.sum(exp)

        idx = int(np.argmax(probs))
        label = self.class_names[idx]
        conf = float(probs[idx])  # 0.0 – 1.0

        with self.lock:
            self.latest_frame = frame.copy()
            self.latest_label = label
            self.latest_conf = conf

    # ---------------- Thread loop ----------------
    def run(self):
        self.running = True
        while self.running:
            self.classify_once()
            time.sleep(0.01)

    # ---------------- Get data ----------------
    def get_data(self):
        with self.lock:
            if self.latest_frame is None:
                return None, None, 0.0
            return self.latest_frame.copy(), self.latest_label, self.latest_conf

    # ---------------- Cleanup ----------------
    def stop(self):
        self.running = False

    def release(self):
        self.cap.release()
