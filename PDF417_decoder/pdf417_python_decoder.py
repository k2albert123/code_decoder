import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pyzbar.pyzbar import decode
import pylibdmtx.pylibdmtx as dmtx

def enhance_image(image):
    """Apply various image enhancement techniques to improve PDF417 detection."""
    enhanced_images = []
    
    # Original image
    enhanced_images.append(("Original", image))
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    enhanced_images.append(("Grayscale", gray))
    
    # Apply Otsu's thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_images.append(("Otsu", otsu))
    
    # Apply adaptive thresholding
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 91, 11)
    enhanced_images.append(("Adaptive", adaptive))
    
    # Apply blur and threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, blur_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_images.append(("Blur+Threshold", blur_thresh))
    
    # Apply sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    enhanced_images.append(("Sharpened", sharpened))
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    enhanced_images.append(("Contrast", contrast))
    
    # Resize image (sometimes helps with detection)
    height, width = gray.shape
    resized = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    enhanced_images.append(("Resized", resized))
    
    # Invert image
    inverted = cv2.bitwise_not(gray)
    enhanced_images.append(("Inverted", inverted))
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    enhanced_images.append(("Dilated", dilated))
    
    eroded = cv2.erode(gray, kernel, iterations=1)
    enhanced_images.append(("Eroded", eroded))
    
    return enhanced_images

def detect_pdf417(image_path, display=True):
    """
    Detect and decode PDF417 barcodes in an image using multiple methods.
    
    Args:
        image_path: Path to the image file
        display: Whether to display the results
        
    Returns:
        Tuple of (decoded_data, points)
    """
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found!")
        return None, None
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image {image_path}")
        return None, None
    
    print(f"Processing image: {image_path}")
    
    # Apply various image enhancements
    enhanced_images = enhance_image(image)
    
    # Try to decode with each enhanced image
    for name, img in enhanced_images:
        print(f"Trying to decode with {name} preprocessing...")
        
        # Try with pyzbar
        try:
            decoded_objects = decode(img)
            if decoded_objects:
                for obj in decoded_objects:
                    if obj.type == 'PDF417':
                        print(f"Decoded PDF417 with {name} preprocessing using pyzbar:")
                        print(f"Data: {obj.data.decode('utf-8')}")
                        
                        # Draw the barcode location
                        points = obj.polygon
                        if len(points) > 0:
                            # Convert points to numpy array for drawing
                            pts = np.array(points, np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            
                            # Draw polygon
                            result_img = image.copy()
                            cv2.polylines(result_img, [pts], True, (0, 255, 0), 3)
                            
                            # Add text with decoded data
                            cv2.putText(result_img, obj.data.decode('utf-8'), 
                                      (points[0][0], points[0][1] - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                            # Save the result
                            result_path = os.path.splitext(image_path)[0] + f"_detected_{name}.png"
                            cv2.imwrite(result_path, result_img)
                            print(f"Result saved to {result_path}")
                            
                            # Display the result
                            if display:
                                plt.figure(figsize=(10, 8))
                                plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                                plt.title(f"Detected PDF417 ({name})")
                                plt.axis('off')
                                plt.show()
                            
                            return obj.data.decode('utf-8'), points
        except Exception as e:
            print(f"Error with pyzbar on {name}: {e}")
        
        # Try with OpenCV's barcode detector (if available in your OpenCV version)
        try:
            if hasattr(cv2, 'barcode') and hasattr(cv2.barcode, 'BarcodeDetector'):
                detector = cv2.barcode.BarcodeDetector()
                retval, decoded_info, decoded_type, points = detector.detectAndDecode(img)
                
                if retval:
                    for i, info in enumerate(decoded_info):
                        if info and len(info) > 0:
                            print(f"Decoded barcode with {name} preprocessing using OpenCV:")
                            print(f"Data: {info}")
                            
                            # Draw the barcode location
                            result_img = image.copy()
                            pts = points[i].astype(np.int32)
                            cv2.polylines(result_img, [pts], True, (0, 255, 0), 3)
                            
                            # Add text with decoded data
                            cv2.putText(result_img, info, 
                                      (int(pts[0][0]), int(pts[0][1]) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                            # Save the result
                            result_path = os.path.splitext(image_path)[0] + f"_detected_opencv_{name}.png"
                            cv2.imwrite(result_path, result_img)
                            print(f"Result saved to {result_path}")
                            
                            # Display the result
                            if display:
                                plt.figure(figsize=(10, 8))
                                plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                                plt.title(f"Detected PDF417 (OpenCV - {name})")
                                plt.axis('off')
                                plt.show()
                            
                            return info, pts
        except Exception as e:
            print(f"Error with OpenCV barcode detector on {name}: {e}")
        
        # Try with pylibdmtx (sometimes works for PDF417)
        try:
            decoded_objects = dmtx.decode(img)
            if decoded_objects:
                for obj in decoded_objects:
                    print(f"Decoded with {name} preprocessing using pylibdmtx:")
                    data = obj.data.decode('utf-8')
                    print(f"Data: {data}")
                    
                    # pylibdmtx doesn't provide polygon points, so we'll just highlight the region
                    result_img = image.copy()
                    x, y = obj.rect.left, obj.rect.top
                    w, h = obj.rect.width, obj.rect.height
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    
                    # Add text with decoded data
                    cv2.putText(result_img, data, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Save the result
                    result_path = os.path.splitext(image_path)[0] + f"_detected_dmtx_{name}.png"
                    cv2.imwrite(result_path, result_img)
                    print(f"Result saved to {result_path}")
                    
                    # Display the result
                    if display:
                        plt.figure(figsize=(10, 8))
                        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                        plt.title(f"Detected Code (pylibdmtx - {name})")
                        plt.axis('off')
                        plt.show()
                    
                    # Create a simple polygon from the rectangle
                    points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                    return data, points
        except Exception as e:
            print(f"Error with pylibdmtx on {name}: {e}")
    
    print("No PDF417 code found after trying multiple preprocessing techniques.")
    print("Tips for better detection:")
    print("1. Ensure the image has good lighting and contrast")
    print("2. Make sure the PDF417 code is clearly visible and not damaged")
    print("3. Try capturing the image from a different angle or with better lighting")
    
    return None, None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect and decode PDF417 barcodes')
    parser.add_argument('--image', '-i', required=True, help='Path to the image containing PDF417 code')
    parser.add_argument('--no-display', action='store_true', help='Do not display the image with detection')
    args = parser.parse_args()
    
    # Detect and decode PDF417
    data, points = detect_pdf417(args.image, not args.no_display)
    
    if data:
        print(f"\nSuccessfully decoded PDF417 barcode:")
        print(f"Data: {data}")
        sys.exit(0)
    else:
        print("\nFailed to decode PDF417 barcode.")
        sys.exit(1)

if __name__ == "__main__":
    main()