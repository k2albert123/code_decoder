import cv2
import numpy as np
import os
import argparse
from pyzbar.pyzbar import decode

def enhance_image(image):
    """Apply various image enhancement techniques to improve barcode detection"""
    enhanced_images = []
    
    # Original image
    enhanced_images.append(("Original", image))
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    enhanced_images.append(("Grayscale", gray))
    
    # Apply adaptive thresholding
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 91, 11)
    enhanced_images.append(("Adaptive", adaptive))
    
    # Apply Otsu's thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_images.append(("Otsu", otsu))
    
    # Sharpen the image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    enhanced_images.append(("Sharpened", sharpened))
    
    # Invert the image
    inverted = cv2.bitwise_not(gray)
    enhanced_images.append(("Inverted", inverted))
    
    return enhanced_images

def detect_pdf417(image_path):
    """Detect and decode PDF417 barcodes in an image using multiple methods"""
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
        
        try:
            decoded_objects = decode(img)
            if decoded_objects:
                for obj in decoded_objects:
                    if obj.type == 'PDF417':
                        print(f"Decoded PDF417 with {name} preprocessing:")
                        print(f"Data: {obj.data.decode('utf-8')}")
                        return obj.data.decode('utf-8'), obj.polygon
                    else:
                        print(f"Found barcode of type {obj.type}, but not PDF417")
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    print("No PDF417 code found in the image after trying multiple preprocessing techniques.")
    return None, None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect and decode PDF417 barcodes')
    parser.add_argument('--image', '-i', required=True, help='Path to the image containing PDF417 code')
    args = parser.parse_args()
    
    # Detect and decode PDF417
    data, points = detect_pdf417(args.image)
    
    if data and points:
        print(f"\nSuccessfully decoded PDF417 barcode: {data}")
        
        # Load the image with OpenCV
        image = cv2.imread(args.image)
        
        # Draw a polygon connecting the points
        points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        print(f"Drawing polygon with points: {points}")
        
        cv2.polylines(image, [points_array], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Save the annotated image
        annotated_image_path = os.path.splitext(args.image)[0] + "_annotated.png"
        cv2.imwrite(annotated_image_path, image)
        print(f"Annotated image saved as {annotated_image_path}")
        
        # Display the image
        cv2.imshow("Detected PDF417", image)
        print("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\nFailed to decode PDF417 barcode.")
        print("Tips for better detection:")
        print("1. Ensure the image has good lighting and contrast")
        print("2. Make sure the PDF417 code is clearly visible and not damaged")
        print("3. Try capturing the image from a different angle or with better lighting")

if __name__ == "__main__":
    main()