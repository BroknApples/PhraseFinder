"""

TODO

"""


if __name__ == "__main__":
  raise RuntimeError("Do not run this script directly.")


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # Silence some tensorflow output
import tensorflow as tf
tf.get_logger().setLevel("ERROR") # Silence more tensorflow output

from typing import Final
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore



# ---------- Constants ----------
IMG_HEIGHT: Final = 32
IMG_WIDTH: Final = 256


def _buildTextClassifier(num_classes: int):
  """
  Docstring for _buildTextClassifier
  
  :param num_words: Description
  :type num_words: int
  """
  
  model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(num_classes, activation='softmax')
  ])

  model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
  )

  return model


def trainModel(data_dir: str) -> bool:
  """
  Docstring for trainModel
  
  :param data_dir: Description
  :type data_dir: str
  :return: Description
  :rtype: bool
  """
  
  # ---------------- Constants ---------------
  
  # Data directory path
  DATA_DIRECTORY_PATH: Final = "data/"
  TRAINING_DIR_PATH: Final = "train/"
  VALIDATION_DIR_PATH: Final = "val/"
  MODELS_DIR_PATH: Final = "models/"
  
  # Epoch count
  NUM_EPOCHS: Final = 20

  # -------------- Build Paths --------------
  if not data_dir.startswith(DATA_DIRECTORY_PATH):
    print(f"Renaming {data_dir}", end='')
    data_dir = os.path.join(DATA_DIRECTORY_PATH, data_dir)
    print(f" to {data_dir}...")
  
  # Ensure directories are valid
  train_dir = os.path.join(data_dir, TRAINING_DIR_PATH)
  if not os.path.exists(train_dir):
    print(f"'{data_dir}' must include '{TRAINING_DIR_PATH}'")
    return False
  val_dir = os.path.join(data_dir, VALIDATION_DIR_PATH)
  if not os.path.exists(val_dir):
    print(f"'{data_dir}' must include '{VALIDATION_DIR_PATH}'")
    return False

  # -------------- Build Dataset --------------
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    label_mode="int"
  )

  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    label_mode="int"
  )
  
  data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
  ])

  # Apply data augmentation to the training dataset
  train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

  # Number of words = number of folders
  num_classes = len(train_ds.class_names)
  print(f"Detected [{num_classes}] classes: {train_ds.class_names}")

  # For speed
  AUTOTUNE = tf.data.AUTOTUNE
  train_ds = train_ds.prefetch(AUTOTUNE)
  val_ds   = val_ds.prefetch(AUTOTUNE)

  # ---------- Train ----------
  model = _buildTextClassifier(num_classes)

  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS
  )

  # ---------- Save model ----------
  filename = os.path.basename(data_dir).strip('/') + ".keras"
  model.save(os.path.join(MODELS_DIR_PATH, filename))

  print(f"Training complete. Model saved as {filename}\n")
  return True
