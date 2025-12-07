#!/bin/sh
# Change working directory to the folder where this script resides
cd "$(dirname "$0")" || exit 1

# Set the script path, images directory, and model path
SCRIPT="../src/text_detect.py"
MODEL="../models/some_model_path"

# Run the script on text images
python3 "$SCRIPT" "<path1>" "$MODEL"
python3 "$SCRIPT" "<path2>" "$MODEL"
python3 "$SCRIPT" "<path3>" "$MODEL"
python3 "$SCRIPT" "<path4>" "$MODEL"
python3 "$SCRIPT" "<path5>" "$MODEL"
