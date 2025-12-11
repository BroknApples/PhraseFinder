# WordDetector
**Author:** Trenton Smith (cvi16)

---

## Overview
WordDetector detects and recognizes text in images using the **EAST text detector** combined with either:

1. **EasyOCR** (recommended, highly accurate)  
2. **Custom CRNN** (pretrained, less accurate)

The project also supports generating synthetic datasets for CRNN training.

---

## Setup

**Known working Python versions:**
- 3.10.11
- 3.11.5

**Install dependencies in a virtual environment:**
```bash
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

> ⚠️ Make sure to activate the virtual environment before running any scripts.

---

## Generating a Dataset (Optional)
> ⚠️ Important: Use a separate virtual environment due to TRDG & EasyOCR conflicts.

1. Install TRDG & compatible pillow version:
```bash
pip install trdg
pip install Pillow==9.5.0
```

2. Create a dictionary file (`dictionary.txt`) listing the words to generate.

3. Generate images for the dictionary:
```bash
python scripts/gen_dataset.py <dictionary_name.txt> <output_directory> <images_per_word>
```

*Example dataset output:*  
![Dataset Example](readme_data\data_gen.png)

---

## Training the Custom CRNN
> ⚠️ Skip this step if you are using EasyOCR.

1. Make sure your dataset directory contains `train/` and `val/` folders structured like:
```
data_dir/
├── train/
│   ├── word1/
│   └── ...
└── val/
    ├── word1/
    └── ...
```

2. Train the CRNN:
```bash
python scripts/train_crnn.py <data_directory> <model_name>
```

- Training may take several hours.  
- Increasing epochs may improve accuracy.  
- The training process uses EarlyStopping and saves the **best model weights**.

*Example training loss graph:*  
![Training Loss](readme_data\training_loss.png)

---

## Running the Text Detector

**Using EasyOCR (Recommended):**
```bash
python scripts/text_detect.py image1.png easyocr 1 true
```

**Using Custom CRNN:**
```bash
python scripts/text_detect.py image1.png crnn 1 true
```

- The last argument (`true`) specifies whether to display the results in a window.  
- The second argument specifies which model to use: `easyocr` or `crnn`.

*Example detection output:*  
![Detection Example](path_to_image.png)

---

## Notes

- The custom CRNN comes pre-trained with `val_loss=0.35`.  
- Dataset generation is optional but required if you want to train your own CRNN.  
- EasyOCR provides significantly better accuracy and is easier to use.

---

## Examples

*Sample detection results:*  
![Example 1](path_to_image.png)  
![Example 2](path_to_image.png)