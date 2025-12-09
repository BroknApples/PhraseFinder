#!/bin/sh
# Change working directory to the folder where this script resides
cd "$(dirname "$0")" || exit 1

# Set the script path, images directory, and model path
SCRIPT="../../src/text_detect.py"
IMG_DIR="<some_image_dirpath>"
MODEL="models/<some_model_path>"

# Run the script on each file in the directory
for f in "$IMG_DIR"/*; do
    python3 "$SCRIPT" "$f" "$MODEL"
done
