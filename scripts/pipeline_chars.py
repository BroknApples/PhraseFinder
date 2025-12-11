# pipeline_chars.py (MODIFIED TO USE EASYOCR)

import sys
import os
import cv2
import numpy as np
import easyocr
# Add project root to sys.path so "src" can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.east_text_detection import detectTextBBFromImage   # Keep this

# CONFIG
IMG_PATH = "image1.png"
RESIZE = 1      # pass to your detector
DISPLAY = True  # show image at end
MIN_CONFIDENCE = 0.50 # General confidence threshold for EasyOCR results

# Initialize EasyOCR Reader once
# Use English ('en') and set GPU=False if you don't have one
reader = easyocr.Reader(['en'], gpu=False) 


def sort_boxes_reading_order(boxes, y_thresh=12):
  # group into lines by y coordinate, then sort within line by x
  if not boxes:
    return []
  boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
  lines = []
  current_line = [boxes_sorted[0]]
  for b in boxes_sorted[1:]:
    if abs(b[1] - current_line[0][1]) <= y_thresh:
      current_line.append(b)
    else:
      lines.append(sorted(current_line, key=lambda bb: bb[0]))
      current_line = [b]
  lines.append(sorted(current_line, key=lambda bb: bb[0]))
  
  # Flatten list of lists back into a single list
  res = [item for sublist in lines for item in sublist]
  return res


def run(image_path=IMG_PATH):
  img = cv2.imread(image_path)
  if img is None:
    raise FileNotFoundError(image_path)

  # 1. Text Detection (EAST)
  boxes = detectTextBBFromImage(img, RESIZE)
  boxes = sort_boxes_reading_order(boxes)

  results = []

  for box in boxes:
    x1, y1, x2, y2 = box
    # pad a bit
    pad = 2
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(img.shape[1], x2 + pad)
    y2p = min(img.shape[0], y2 + pad)
    crop = img[y1p:y2p, x1p:x2p]

    # Recognition with detail=1 (bbox, text, confidence)
    recognized_segments = reader.readtext(
        crop,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
        detail=1
    )

    word = ""
    confidences = []

    # Sort segments left-to-right to avoid merging issues
    recognized_segments.sort(key=lambda s: s[0][0][0])
    for bbox, text, conf in recognized_segments:
      if conf >= MIN_CONFIDENCE:
        word += text
        confidences.append(conf)

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    results.append((box, word, avg_conf))

  # print and draw
  out = img.copy()
  for (box, word, avg_conf) in results:
    x1, y1, x2, y2 = box
    
    # Skip if word is empty or confidence is too low (optional filtering)
    if not word or avg_conf < MIN_CONFIDENCE:
        continue
        
    print(f"[{word}] (Conf: {avg_conf:.2f})")

    # Draw box and text
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(out, word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

  if DISPLAY:
    cv2.imshow("EasyOCR Result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
  run()