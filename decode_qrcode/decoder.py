import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode as pyzbar_decode

def decode_qrcode(image_path):
    """
    Decode QR code from an image with enhanced preprocessing.
    
    Args:
        image_path (str): Path to the image containing the QR code.
        
    Returns:
        str: Decoded data from the QR code.
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
        
        # Try both OpenCV's QR detector and pyzbar
        for name, processed_img in preprocessed_images:
            # Try OpenCV QR detector
            qr_detector = cv2.QRCodeDetector()
            data, bbox, _ = qr_detector.detectAndDecode(processed_img if len(processed_img.shape) == 3 else cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR))
            
            if data:
                print(f"Decoded QR Code using OpenCV ({name}): {data}")
                
                # Display the image with the detected QR code
                if bbox is not None:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    plt.title(f"Detected QR Code (OpenCV - {name})")
                    
                    # Draw polygon around the QR code
                    bbox = bbox.astype(int)
                    n = len(bbox[0])
                    for i in range(n):
                        cv2.line(image, tuple(bbox[0][i]), tuple(bbox[0][(i+1) % n]), (0,255,0), 3)
                    
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    plt.savefig(image_path.replace('.png', f'_detected_opencv_{name}.png'))
                    plt.close()
                
                return data
            
            # Try pyzbar
            try:
                decoded_objects = pyzbar_decode(processed_img)
                for obj in decoded_objects:
                    if obj.type == 'QRCODE':
                        data = obj.data.decode('utf-8')
                        print(f"Decoded QR Code using pyzbar ({name}): {data}")
                        
                        # Display the image with the detected QR code
                        plt.figure(figsize=(10, 8))
                        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        plt.title(f"Detected QR Code (pyzbar - {name})")
                        
                        # Draw rectangle around the QR code
                        rect = obj.rect
                        plt.plot([rect.left, rect.left, rect.left + rect.width, rect.left + rect.width, rect.left],
                                [rect.top, rect.top + rect.height, rect.top + rect.height, rect.top, rect.top], 'r-')
                        
                        plt.savefig(image_path.replace('.png', f'_detected_pyzbar_{name}.png'))
                        plt.close()
                        
                        return data
            except Exception as e:
                print(f"pyzbar error with {name}: {e}")
                continue
        
        print("No QR code found in the image after trying multiple preprocessing techniques.")
        return None
            
    except Exception as e:
        print(f"Error decoding QR code: {e}")
        return None

if __name__ == "__main__":
    # Path to the image containing the QR code
    image_path = "image.png"
    
    # Decode the QR code
    decoded_data = decode_qrcode(image_path)
    
    if decoded_data:
        print(f"Decoded data: {decoded_data}")
    else:
        print("Failed to decode QR code.")
        print("Note: For better QR code detection, ensure the image has good lighting and contrast.")
        print("You may need to install pyzbar: pip install pyzbar")