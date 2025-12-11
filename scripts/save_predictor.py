"""
Create Prediction Model from saved CRNN weights
"""

import sys
import os
from tensorflow import keras
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras import layers # type: ignore

# -------------------- Constants --------------------
IMG_HEIGHT = 32
IMG_WIDTH = 256
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,-'"
MAX_LABEL_LEN = 16
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank

WEIGHTS_PATH = "models/crnn.weights.h5"        # Your saved weights
PREDICTOR_PATH = "models/crnn_predictor.keras"  # Where to save the predictor


# -------------------- Build Prediction Model --------------------
def buildPredictionModel():
  input_img = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")

  # Conv layers
  x = layers.Conv2D(64, 3, padding="same", activation="relu")(input_img)
  x = layers.MaxPooling2D((2,2))(x)
  x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
  x = layers.MaxPooling2D((2,2))(x)

  # Collapse height
  pool_h, pool_w = x.shape[1], x.shape[2]
  x = layers.Reshape(target_shape=(pool_w, pool_h * 128))(x)

  # BiLSTM
  x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
  x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)

  # Output
  y_pred = layers.Dense(NUM_CLASSES, activation="softmax", name="y_pred")(x)

  model = keras.Model(inputs=input_img, outputs=y_pred, name="crnn_predictor")
  return model

if __name__ == "__main__":
  # -------------------- Load Weights and Save Predictor --------------------
  if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Weights file not found at {WEIGHTS_PATH}")

  predictor = buildPredictionModel()
  predictor.load_weights(WEIGHTS_PATH)
  predictor.save(PREDICTOR_PATH)

  print(f"Prediction model saved at: {PREDICTOR_PATH}")
