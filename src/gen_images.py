"""
should be called like

python src/gen_images.py "path_to_input_file(contain a list of phrases/characters to detect, separated per-line)" "output_directory_name(will be in the 'data/' directory)"
"""

import sys
import os
from typing import Final
from trdg.generators import (
  GeneratorFromDict,
  GeneratorFromRandom,
  GeneratorFromStrings,
  GeneratorFromWikipedia,
)

# ********************** beg Functions ********************** #

def main(argc: int, argv: list[str]) -> int:
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

  # Bounds checks
  if argc != 4:
    print(f"Usage: {argv[0]} <dict_path.txt> <output_dirname> <count>")
    return -1

  # ---------------- Constants ---------------

  # Path to the dictionary directory
  DICT_DIRECTORY_PATH: Final = "dict/"

  # Path to the dictionary directory
  DATA_DIRECTORY_PATH: Final = "data/"

  # ------------- Ensure proper names -------------

  # Get command line args and check paths
  dict_path = argv[1]
  output_dirname = argv[2]
  count = argv[3]

  # Ensure the paths start with the proper prefix
  if not dict_path.startswith(DICT_DIRECTORY_PATH):
    print(f"Renaming {dict_path}", end='')
    dict_path = os.path.join(DICT_DIRECTORY_PATH, dict_path)
    print(f" to {dict_path}...")
  if not output_dirname.startswith(DATA_DIRECTORY_PATH):
    print(f"Renaming {output_dirname}", end='')
    output_dirname = os.path.join(DATA_DIRECTORY_PATH, output_dirname)
    print(f" to {output_dirname}...")

  # Checks if the dictionary file exists
  if not os.path.exists(dict_path):
    print(f"Input file [f{dict_path}] does not exist.")
    return -1

  # Checks if the output directory is valid (empty or non-existent)
  if os.path.exists(output_dirname) and bool(os.listdir(output_dirname)):
    print(f"Output directory [{output_dirname}] already exists with files, please choose a non-existent or empty directory.")
    return -1

  # ------------- Gather phrases and run trdg -------------
  phrases: str = []
  with open(dict_path, "r") as f:
    phrases = f.readlines()  # includes newline characters

  return 0

# ********************** end Functions ********************** #

# Run automatically if ran as the main script.
if __name__ == "__main__":
  ret: int = main(len(sys.argv), sys.argv)

  if ret == 0:
    print("Operation completed successfully!")
  elif ret == -1:
    print("Operation did no complete properly. Error detected.")
