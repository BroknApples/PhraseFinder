"""

"""

import cv2
import numpy as np
from typing import Final

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # Silence some tensorflow output
import tensorflow as tf
tf.get_logger().setLevel("ERROR") # Silence more tensorflow output

from typing import Final
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras import backend as K # type: ignore



# ---------- Constants ----------
IMG_HEIGHT: Final = 32
IMG_WIDTH: Final = 256
CHARS: Final = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,-'"
NUM_CLASSES: Final = len(CHARS) + 1  # +1 for CTC blank


# ---------- Helper functions ----------
def _preprocessBox(image, box):
  x1, y1, x2, y2 = box
  crop = image[y1:y2, x1:x2]
  crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
  crop = cv2.resize(crop, (IMG_WIDTH, IMG_HEIGHT))  # fixed width
  crop = crop.astype("float32") / 255.0
  crop = np.expand_dims(crop, axis=(0, -1))
  return crop

def _decodePrediction(pred, chars=CHARS):
  # pred: [batch, time_steps, num_classes]
  input_len = np.ones(pred.shape[0]) * pred.shape[1]
  decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
  decoded = decoded[0].numpy()

  result = []
  for seq in decoded:
    text = ""
    for c in seq:
      if c == -1: continue # Blank
      elif c >= len(chars): continue
      text += chars[c]
    result.append(text)
  return result


def _sortBoxes(boxes, threshold=10):
  """
  Sorts boxes by reading order: top to bottom, then left to right.
  threshold: max y-distance to consider boxes on the same line
  """

  # Sort by y-coordinate
  boxes = sorted(boxes, key=lambda b: b[1])

  sorted_boxes = []
  curr_line = []

  for box in boxes:
    if not curr_line:
      curr_line.append(box)
      continue

    # This box is on the same line
    if abs(box[1] - curr_line[0][1]) < threshold:
      curr_line.append(box)
    else:
      # Sort line left to right
      curr_line.sort(key=lambda b: b[0])
      sorted_boxes.extend(curr_line)
      curr_line = [box]

  # Add the last line
  if curr_line:
    curr_line.sort(key=lambda b: b[0])
    sorted_boxes.extend(curr_line)

  return sorted_boxes


def recognizeTextFromBBs(image: np.ndarray, boxes: list[tuple[int, int, int, int]], model):
  """
  Docstring for recognizeTextFromBBs
  
  :param image: Description
  :param boxes: Description
  :param model: Description
  """

  boxes = _sortBoxes(boxes)

  results = []
  for box in boxes:
    crop = _preprocessBox(image, box)
    pred = model.predict(crop)
    text = _decodePrediction(pred)
    if text:
        results.append((box, text))
  
  return results