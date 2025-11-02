import cv2
import numpy as np
import os
import argparse
from pyzbar.pyzbar import decode

def preprocess_image(image):
    """Apply basic preprocessing to improve barcode detection"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 91, 11)
    return thresh

def detect_pdf417(image_path):
    """Detect and decode PDF417 barcodes in an image"""
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found!")
        return None, None
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image {image_path}")
        return None, None
    
    # Try to decode with original image
    decoded_objects = decode(image)
    if decoded_objects:
        for obj in decoded_objects:
            if obj.type == 'PDF417':
                print(f"Decoded PDF417 from original image:")
                print(f"Data: {obj.data.decode('utf-8')}")
                return obj.data.decode('utf-8'), obj.polygon
    
    # Preprocess the image and try again
    processed = preprocess_image(image)
    decoded_objects = decode(processed)
    if decoded_objects:
        for obj in decoded_objects:
            if obj.type == 'PDF417':
                print(f"Decoded PDF417 after preprocessing:")
                print(f"Data: {obj.data.decode('utf-8')}")
                return obj.data.decode('utf-8'), obj.polygon
    
    print("No PDF417 code found in the image.")
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

if __name__ == "__main__":
    main()