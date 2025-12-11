"""


Usage: python scripts/text_detect.py <image_path>
"""


import sys
import os

# Add project root to sys.path so "src" can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.crnn_training import (
  train_crnn
)


def main(argc: int, argv: list[str]) -> int:
  """
  Entry point for generating synthetic text images from a phrase list.

  Parses command-line arguments, validates input/output paths, reads
  phrases from a dictionary file, and runs multiple TRDG batches for
  test and validation sets.

  Parameters
  ----------
    argc : Number of command-line arguments provided.
    argv : List of command-line arguments. Expected order:
             
             0: script name
             
             1: The image size to be taken by the EAST detector (0 = (320, 320)), (1 = (512, 512)), (2 = (640, 640))

  Returns
  -------
    int : Status code: 0 on success, -1 if an error \
          occurred (e.g., invalid arguments or paths).
  """

  if argc < 1 or argc > 3: # Must be 0-2 arguments
    print("Usage: python train_cnn.py <data_directory> [plot_history]")
    return -1
  
  model_name = argv[1] if argc > 1 else "text_classifier"
  plot_history = (argv[2].lower() in ("1", "true", "yes")) if argc > 2 else False

  if not train_crnn(model_name, "crnn", plot_history):
    return -1
  
  return 0


# Run automatically if ran as the main script.
if __name__ == "__main__":
  ret: int = main(len(sys.argv), sys.argv)

  if ret == 0:
    print("Operation completed successfully!")
  elif ret == -1:
    print("Operation did no complete properly. Error detected.")
