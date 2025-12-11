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


# ---------- Constants ----------
IMG_HEIGHT: Final = 32
IMG_WIDTH: Final = 256

def preprocess_image(img_path, img_height=32, img_width=256):
  img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  h, w = img.shape
  # Resize to fixed height, keep aspect ratio
  new_w = int(img_width * w / h)
  img = cv2.resize(img, (new_w, img_height))
  # Pad width to fixed width
  if new_w < img_width:
      pad = np.zeros((img_height, img_width - new_w), dtype=np.uint8)
      img = np.hstack((img, pad))
  # Normalize
  img = img.astype("float32") / 255.0
  # Add batch & channel dimension
  img = np.expand_dims(img, axis=(0,-1))
  return img

from tensorflow.keras.models import load_model # type: ignore
import tensorflow.keras.backend as K # type: ignore

def decode_prediction(pred, chars="abcdefghijklmnopqrstuvwxyz0123456789"):
    decoded, _ = K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], greedy=True)
    out_best = decoded[0].numpy()
    decoded_text = ""
    for c in out_best[0]:
        if c == len(chars):  # CTC blank
            continue
        decoded_text += chars[c]
    return decoded_text
