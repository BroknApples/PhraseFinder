"""
Usage: python scripts/text_detect.py <image_path> <model_path> [resize] [display_bb]
"""

import sys
import os
import cv2
import numpy as np
import tensorflow as tf

# Add project root to sys.path so "src" can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from typing import Final
from tensorflow.keras.models import load_model # type: ignore

from src.east_text_detection import detectTextBBFromImage, detectTextBBFromImages
from src.crnn_text_identifaction import recognizeTextFromBBs


# ---------- Main ----------
def main(argc: int, argv: list[str]) -> int:
  """
  Docstring for main
  
  :param argc: Description
  :type argc: int
  :param argv: Description
  :type argv: list[str]
  :return: Description
  :rtype: int
  """

  if argc < 3 or argc > 5:
    print("Usage: python text_detect.py <image_path> <model_path> [resize] [display_bb]")
    return -1

  image_path = argv[1]
  model_path = argv[2]
  resize = int(argv[3]) if len(argv) > 3 else 1
  display_bb = (argv[4].lower() in ("1", "true", "yes")) if len(argv) > 4 else False

  if not os.path.exists(image_path):
    print(f"Invalid image path [{image_path}]")
    return -1
  if not os.path.exists(model_path):
    print(f"Invalid model path [{model_path}]")
    return -1

  # Load prediction model (CRNN without Lambda)
  model = load_model(model_path, compile=False)

  # Load image
  image = cv2.imread(image_path, cv2.IMREAD_COLOR)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Detect text boxes
  boxes = detectTextBBFromImage(image, resize, display_bb)

  results = recognizeTextFromBBs(image, boxes, model)

  # Display results
  for tup in results:
    (x1, y1, x2, y2), word = tup
    print(f"{(x1, y1, x2, y2)} : {word}")

  copy = image.copy()
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.5
  font_thickness = 1
  
  for tup in results:
    (x1, y1, x2, y2), word_list = tup # Rename variable to be clear it's a list
    
    # --- FIX IS HERE ---
    word_string = word_list[0] # Take the string out of the list
    
    # 1. Draw Bounding Box (Existing Code)
    cv2.rectangle(copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 2. Draw Text
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > 10 else y2 + 20 
    
    cv2.putText(
      copy, 
      word_string, # Use the string variable
      (text_x, text_y), 
      font, 
      font_scale, 
      (255, 0, 0), 
      font_thickness, 
      cv2.LINE_AA
    )

  # 3. Show Final Image (New Code)
  # Convert back to BGR for display/save if image started as BGR
  # Assuming 'image' was read in BGR originally:
  copy_bgr = cv2.cvtColor(copy, cv2.COLOR_RGB2BGR) 
  
  cv2.imshow("Detection & Recognition Results", copy_bgr)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  return 0


# Run automatically if ran as the main script
if __name__ == "__main__":
  ret: int = main(len(sys.argv), sys.argv)
  if ret == 0:
    print("Operation completed successfully!")
  else:
    print("Operation did not complete properly. Error detected.")
