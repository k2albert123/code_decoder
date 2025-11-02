import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode

def decode_pdf417(image_path):
    """
    Decode PDF417 code from an image with enhanced preprocessing.
    
    Args:
        image_path (str): Path to the image containing the PDF417 code.
        
    Returns:
        str: Decoded data from the PDF417 code.
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        # Try multiple preprocessing techniques
        preprocessed_images = []
        
        # Original image
        preprocessed_images.append(("Original", image))
        
        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessed_images.append(("Grayscale", gray))
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 91, 11)
        preprocessed_images.append(("Adaptive Threshold", thresh))
        
        # Otsu's thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("Otsu Threshold", otsu))
        
        # Blur and threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, blur_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("Blur + Threshold", blur_thresh))
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        preprocessed_images.append(("Sharpened", sharpened))
        
        # Try decoding with each preprocessing technique
        for name, processed_img in preprocessed_images:
            try:
                # Decode the PDF417 code
                decoded_objects = decode(processed_img)
                
                for obj in decoded_objects:
                    if obj.type == 'PDF417':
                        data = obj.data.decode('utf-8')
                        print(f"Decoded PDF417 Code ({name}): {data}")
                        
                        # Display the image with the detected PDF417 code
                        plt.figure(figsize=(10, 8))
                        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        plt.title(f"Detected PDF417 Code ({name})")
                        
                        # Draw rectangle around the PDF417 code
                        rect = obj.rect
                        cv2.rectangle(image, (rect.left, rect.top), 
                                    (rect.left + rect.width, rect.top + rect.height), 
                                    (0, 255, 0), 3)
                        
                        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        plt.savefig(image_path.replace('.png', f'_detected_{name}.png'))
                        plt.close()
                        
                        return data
            except Exception as e:
                print(f"Error with {name} preprocessing: {e}")
                continue
        
        # Try additional preprocessing combinations
        try:
            # Resize image (2x)
            resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            decoded_objects = decode(gray_resized)
            
            for obj in decoded_objects:
                if obj.type == 'PDF417':
                    data = obj.data.decode('utf-8')
                    print(f"Decoded PDF417 Code (Resized 2x): {data}")
                    return data
                    
            # Invert image
            inverted = cv2.bitwise_not(gray)
            decoded_objects = decode(inverted)
            
            for obj in decoded_objects:
                if obj.type == 'PDF417':
                    data = obj.data.decode('utf-8')
                    print(f"Decoded PDF417 Code (Inverted): {data}")
                    return data
                    
            # Try morphological operations
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(gray, kernel, iterations=1)
            decoded_objects = decode(dilated)
            
            for obj in decoded_objects:
                if obj.type == 'PDF417':
                    data = obj.data.decode('utf-8')
                    print(f"Decoded PDF417 Code (Dilated): {data}")
                    return data
        except Exception as e:
            print(f"Error with additional preprocessing: {e}")
        
        print("No PDF417 code found in the image after trying multiple preprocessing techniques.")
        return None
            
    except Exception as e:
        print(f"Error decoding PDF417 code: {e}")
        print("Note: You may need to install pyzbar: pip install pyzbar")
        print("      You may also need Visual C++ Redistributable: https://aka.ms/vs/16/release/vc_redist.x64.exe")
        return None

if __name__ == "__main__":
    # Path to the image containing the PDF417 code
    image_path = "image.png"
    
    # Decode the PDF417 code
    decoded_data = decode_pdf417(image_path)
    
    if decoded_data:
        print(f"Decoded data: {decoded_data}")
    else:
        print("Failed to decode PDF417 code.")
        print("Note: For better PDF417 code detection, ensure the image has good lighting and contrast.")

