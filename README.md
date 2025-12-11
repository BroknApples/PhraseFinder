# NAME
Author: **Trenton Smith** *- cvi16*


## Overview



## Setup

**KNOWN WORKING PYTHON VERSIONS**
* 3.10.11
* 3.11.5

--------

It is recommended that you use a Python Virtual Environment and install the packages listed in `requirements.txt`

```terminal
python -m venv venv
venv/Scripts/Activate
pip install -r requirements.txt
```


## How to Use

This program has two methods of use.
1. *Run using EAST + EasyOCR*
  (very accurate)
2. *Run using EAST + a custom CRNN*
  (not accurate at all)

This project comes with a premade custom CRNN with `val_loss=0.35`

### Generating a dataset

**NOTE: (VERY IMPORTANT) To do this, you must create a seperetae virtual environment as there are package conflicts with trdg & easyocr**

In you virtual environment, run:
```terminal
pip install trdg
```

Create a dictionary file (a simple text file containing the words you want to generate images for). There is a premade dictionary crnn which trained the premade model.

```terminal
python scripts/gen_dataset.py <dictionary_name.txt> <output_directory> <images to generate per word>
```



### Training the custom CRNN

**NOTE: If you are using EasyOCR for detection, skip to "Running the text detector"**

**NOTE: This may take many hours (a very long time)**

Run the following python command, where data_directory=`path to the folder containing 'data/' and 'val'`
When training, upping the Epoch count will lead to a more reliable CRNN.

```terminal
python scripts/gen_dataset.py <dictionary_name.txt> <output_directory> <images to generate per word>
```

Outputted results
<TODO link image of training loss>

### Running the text detector

*Using EasyOCR (Recommended)*
```terminal
python scripts/text_detect.py image1.png easyocr 1 true
```

OR

*Using the custom CRNN*
```terminal
python scripts/text_detect.py image1.png crnn 1 true
```




## Examples


