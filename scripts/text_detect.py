"""
Usage: python scripts/text_detect.py <image_path> <model_path> [resize] [display_bb]
"""

import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Add project root to sys.path so "src" can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from typing import Final
from src.east_text_detection import detectTextBBFromImage

# ---------- Constants ----------
IMG_HEIGHT: Final = 32
IMG_WIDTH: Final = 256
CHARS: Final = "abcdefghijklmnopqrstuvwxyz0123456789"
NUM_CLASSES: Final = len(CHARS) + 1  # +1 for CTC blank

# ---------- Helper functions ----------
def preprocess_box(image, box):
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (IMG_WIDTH, IMG_HEIGHT))  # fixed width
    crop = crop.astype("float32") / 255.0
    crop = np.expand_dims(crop, axis=(0, -1))
    return crop

def decode_prediction(pred, chars="abcdefghijklmnopqrstuvwxyz0123456789"):
    # pred: [batch, time_steps, num_classes]
    input_len = np.ones(pred.shape[0]) * pred.shape[1]  # all sequences are full length
    decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
    decoded = decoded[0].numpy()  # shape: (batch, max_label_len)

    result = []
    for seq in decoded:
        text = ""
        for c in seq:
            if c == -1:  # CTC blank
                continue
            if c >= len(chars):
                continue
            text += chars[c]
        result.append(text)
    return result


# ---------- Main ----------
def main(argc: int, argv: list[str]) -> int:
    if argc < 3:
        print("Usage: python text_detect.py <image_path> <model_path> [resize] [display_bb]")
        return -1

    image_path = argv[1]
    model_path = argv[2]
    resize = int(argv[3]) if len(argv) > 3 else 1
    display_bb = (argv[4].lower() in ("1", "true", "yes")) if len(argv) > 4 else False

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Invalid image path [{image_path}]")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Invalid model path [{model_path}]")

    # Load prediction model (CRNN without Lambda)
    model = load_model(model_path, compile=False)

    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect text boxes
    boxes = detectTextBBFromImage(image, resize, display_bb)

    results = []
    for box in boxes:
        crop = preprocess_box(image, box)
        pred = model.predict(crop)
        text = decode_prediction(pred)
        if text:
            results.append((box, text))

    # Display results
    copy = image.copy()
    for tup in results:
        (x1, y1, x2, y2), word = tup
        cv2.rectangle(copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"{(x1, y1, x2, y2)} : {word}")

    cv2.imshow("Text Detection", copy)
    cv2.waitKey(0)

    return 0

# Run automatically if ran as the main script
if __name__ == "__main__":
    ret: int = main(len(sys.argv), sys.argv)
    if ret == 0:
        print("Operation completed successfully!")
    else:
        print("Operation did not complete properly. Error detected.")
