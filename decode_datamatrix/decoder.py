import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode
import matplotlib.pyplot as plt

def decode_datamatrix(image_path):
    """
    Decode Data Matrix code from an image.
    
    Args:
        image_path (str): Path to the image containing the Data Matrix code.
        
    Returns:
        str: Decoded data from the Data Matrix code.
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Decode the Data Matrix code
        decoded_objects = decode(gray)
        
        if decoded_objects:
            for obj in decoded_objects:
                data = obj.data.decode('utf-8')
                print(f"Decoded Data Matrix: {data}")
                
                # Display the image with the detected Data Matrix
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title("Detected Data Matrix")
                
                # Draw rectangle around the Data Matrix (if coordinates are available)
                if hasattr(obj, 'rect'):
                    rect = obj.rect
                    plt.plot([rect.left, rect.left, rect.left + rect.width, rect.left + rect.width, rect.left],
                            [rect.top, rect.top + rect.height, rect.top + rect.height, rect.top, rect.top], 'r-')
                
                plt.savefig(image_path.replace('.png', '_detected.png'))
                plt.close()
                
                return data
            
            return None
        else:
            print("No Data Matrix codes detected in the image.")
            return None
            
    except Exception as e:
        print(f"Error decoding Data Matrix: {e}")
        print("Note: Data Matrix decoding requires the pylibdmtx library.")
        print("Install it using: pip install pylibdmtx")
        return None

if __name__ == "__main__":
    # Path to the image containing the Data Matrix code
    image_path = "image.png"
    
    # Decode the Data Matrix code
    decoded_data = decode_datamatrix(image_path)
    
    if decoded_data:
        print(f"Decoded data: {decoded_data}")
    else:
        print("Failed to decode Data Matrix code.")
        print("Note: Data Matrix decoding requires the pylibdmtx library.")
        print("Install it using: pip install pylibdmtx")