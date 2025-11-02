# Aztec Code Decoder

This directory contains a tool for detecting and decoding Aztec codes from images.

## Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- NumPy
- pyzbar
- Matplotlib

## Usage

```bash
python decoder.py
```

The script will:
1. Load the image file named `image.png` from the current directory
2. Apply multiple image preprocessing techniques to improve detection
3. Attempt to decode Aztec codes using pyzbar
4. Display and save the detected Aztec code with a bounding box
5. Output the decoded data

## Features

- Multiple preprocessing techniques for improved detection:
  - Original image
  - Grayscale conversion
  - Adaptive thresholding
  - Otsu's thresholding
  - Blur and threshold
  - Image sharpening
  - Contrast enhancement
  - Image resizing
  - Image inversion

## Troubleshooting

If the decoder fails to detect an Aztec code:

1. Ensure the image has good lighting and contrast
2. Make sure the Aztec code is clearly visible and not damaged
3. Try capturing the image from a different angle or with better lighting
4. Verify that all dependencies are properly installed
5. Ensure Visual C++ Redistributable is installed (required for pyzbar)