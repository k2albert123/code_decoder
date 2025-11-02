# Barcode Decoder

This directory contains a tool for detecting and decoding linear barcodes (EAN, UPC, Code 128, etc.) from images.

## Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- pyzbar
- Matplotlib

## Usage

```bash
python decoder.py
```

The script will:
1. Load the image file named `image.png` from the current directory
2. Decode barcodes using pyzbar
3. Display and save the detected barcode with a bounding box
4. Output the decoded data and barcode type

## Supported Barcode Types

- EAN-13, EAN-8
- UPC-A, UPC-E
- Code 39, Code 128
- And other linear barcode formats supported by pyzbar

## Troubleshooting

If the decoder fails to detect a barcode:
1. Ensure the image has good lighting and contrast
2. Make sure the barcode is clearly visible and not damaged
3. Verify that all dependencies are properly installed