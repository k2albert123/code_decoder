import cv2
import numpy as np
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt

def decode_barcode(image_path):
    """
    Decode barcode from an image.
    
    Args:
        image_path (str): Path to the image containing the barcode.
        
    Returns:
        str: Decoded data from the barcode.
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Decode the barcode
        decoded_objects = decode(gray)
        
        if decoded_objects:
            for obj in decoded_objects:
                print(f"Decoded Barcode ({obj.type}): {obj.data.decode('utf-8')}")
                
                # Display the image with the detected barcode
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title(f"Detected {obj.type} Barcode")
                
                # Draw rectangle around the barcode
                rect = obj.rect
                plt.plot([rect.left, rect.left, rect.left + rect.width, rect.left + rect.width, rect.left],
                        [rect.top, rect.top + rect.height, rect.top + rect.height, rect.top, rect.top], 'r-')
                
                plt.savefig(image_path.replace('.png', '_detected.png'))
                plt.close()
                
                return obj.data.decode('utf-8')
            
            return None
        else:
            print("No barcodes detected in the image.")
            return None
            
    except Exception as e:
        print(f"Error decoding barcode: {e}")
        return None

if __name__ == "__main__":
    # Path to the image containing the barcode
    image_path = "image.png"
    
    # Decode the barcode
    decoded_data = decode_barcode(image_path)
    
    if decoded_data:
        print(f"Decoded data: {decoded_data}")
    else:
        print("Failed to decode barcode.")
        print("Note: Barcode decoding requires the pyzbar library.")
        print("Install it using: pip install pyzbar")
        print("On Windows, you might also need to install Visual C++ Redistributable.")