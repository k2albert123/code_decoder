# PDF417 Barcode Detector

This directory contains tools for detecting and decoding PDF417 barcodes using ZXing and Docker.

## Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- NumPy
- Docker
- Internet connection (for downloading JAR files if not present)

## Files

- `pdf417_detector.py`: Main script for detecting PDF417 barcodes using ZXing with Docker
- `decode_pdf417_zxing.py`: Library file with functions for PDF417 detection
- `decode_pdf417.py`: Original decoder using pyzbar (may have limited success)

## Usage

### Using the PDF417 Detector

```bash
python pdf417_detector.py --image path/to/your/image.png
```

Options:
- `--image` or `-i`: Path to the image containing the PDF417 barcode (required)
- `--no-display`: Do not display the image with detection (optional)

### Example

```bash
python pdf417_detector.py --image id-gabriel.png
```

## How It Works

1. The script checks for required JAR files and downloads them if necessary
2. It preprocesses the image using multiple techniques:
   - Original image
   - Otsu's thresholding
   - Adaptive thresholding
   - Image sharpening
3. It attempts to decode the PDF417 barcode using ZXing with each preprocessed image
4. When a barcode is detected, it draws a green polygon around it and saves the annotated image

## Troubleshooting

If the detector fails to find a PDF417 barcode:

1. Ensure the image has good lighting and contrast
2. Make sure the PDF417 code is clearly visible and not damaged
3. Try capturing the image from a different angle or with better lighting
4. Verify that Docker is installed and running

## Required JAR Files

The following JAR files will be automatically downloaded if not present:
- `javase-3.5.0.jar`
- `core-3.5.0.jar`
- `jcommander-1.82.jar`