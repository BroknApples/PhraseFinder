"""


Usage: python src/text_detect.py <image_path>
"""


import sys
import os
import numpy
import cv2
import torch

from east_text_detection import (
  detectTextBBFromImage,
  detectTextBBFromImages,
)


def main() -> int:
  """
  TODO...

  Parameters
  ----------
    argc : Number of command-line arguments
    argv : List of command-line arguments

  Returns
  -------
    int : operation return value
    
  -------
  """

  # TODO Should use argv[1] to be the path to the image, argv[2] should be the model path
  
  return 0


# Run automatically if ran as the main script.
if __name__ == "__main__":
  ret: int = main(len(sys.argv), sys.argv)

  if ret == 0:
    print("Operation completed successfully!")
  elif ret == -1:
    print("Operation did no complete properly. Error detected.")
