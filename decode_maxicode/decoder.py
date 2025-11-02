import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode

def decode_maxicode(image_path):
    """
    Decode MaxiCode from an image with enhanced preprocessing.
    
    Args:
        image_path (str): Path to the image containing the MaxiCode.
        
    Returns:
        str: Decoded data from the MaxiCode.
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
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(gray)
        preprocessed_images.append(("CLAHE", clahe_img))
        
        # Try decoding with each preprocessing technique
        for name, processed_img in preprocessed_images:
            try:
                # Decode the MaxiCode
                decoded_objects = decode(processed_img)
                
                for obj in decoded_objects:
                    if obj.type == 'MAXICODE':
                        data = obj.data.decode('utf-8')
                        print(f"Decoded MaxiCode ({name}): {data}")
                        
                        # Display the image with the detected MaxiCode
                        plt.figure(figsize=(10, 8))
                        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        plt.title(f"Detected MaxiCode ({name})")
                        
                        # Draw rectangle around the MaxiCode
                        rect = obj.rect
                        plt.plot([rect.left, rect.left, rect.left + rect.width, rect.left + rect.width, rect.left],
                                [rect.top, rect.top + rect.height, rect.top + rect.height, rect.top, rect.top], 'r-')
                        
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
                if obj.type == 'MAXICODE':
                    data = obj.data.decode('utf-8')
                    print(f"Decoded MaxiCode (Resized 2x): {data}")
                    return data
                    
            # Invert image
            inverted = cv2.bitwise_not(gray)
            decoded_objects = decode(inverted)
            
            for obj in decoded_objects:
                if obj.type == 'MAXICODE':
                    data = obj.data.decode('utf-8')
                    print(f"Decoded MaxiCode (Inverted): {data}")
                    return data
                    
            # Try morphological operations
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(gray, kernel, iterations=1)
            decoded_objects = decode(dilated)
            
            for obj in decoded_objects:
                if obj.type == 'MAXICODE':
                    data = obj.data.decode('utf-8')
                    print(f"Decoded MaxiCode (Dilated): {data}")
                    return data
                    
            eroded = cv2.erode(gray, kernel, iterations=1)
            decoded_objects = decode(eroded)
            
            for obj in decoded_objects:
                if obj.type == 'MAXICODE':
                    data = obj.data.decode('utf-8')
                    print(f"Decoded MaxiCode (Eroded): {data}")
                    return data
        except Exception as e:
            print(f"Error with additional preprocessing: {e}")
        
        print("No MaxiCode found in the image after trying multiple preprocessing techniques.")
        return None
            
    except Exception as e:
        print(f"Error decoding MaxiCode: {e}")
        return None

if __name__ == "__main__":
    # Path to the image containing the MaxiCode
    image_path = "image.png"
    
    # Decode the MaxiCode
    decoded_data = decode_maxicode(image_path)
    
    if decoded_data:
        print(f"Decoded data: {decoded_data}")
    else:
        print("Failed to decode MaxiCode.")
        print("Note: MaxiCode decoding requires the pyzbar library.")
        print("Install it using: pip install pyzbar")
        print("On Windows, you might also need to install Visual C++ Redistributable.")