# DataMatrix Decoder

This directory contains a tool for detecting and decoding DataMatrix codes from images.

## Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- pylibdmtx
- Matplotlib

## Usage

```bash
python decoder.py
```

The script will:
1. Load the image file named `image.png` from the current directory
2. Decode DataMatrix codes using pylibdmtx
3. Display and save the detected code with a bounding box
4. Output the decoded data

## Features

- Specialized DataMatrix detection using pylibdmtx library
- Visual output with bounding box around detected codes

## Troubleshooting

If the decoder fails to detect a DataMatrix code:
1. Ensure the image has good lighting and contrast
2. Make sure the DataMatrix code is clearly visible and not damaged
3. Try capturing the image from a different angle or with better lighting