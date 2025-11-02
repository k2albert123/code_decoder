import cv2
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt

def decode_pdf417_zxing(image_path):
    """
    Decode PDF417 code from an image using ZXing with Docker.
    
    Args:
        image_path (str): Path to the image containing the PDF417 code.
        
    Returns:
        str: Decoded data from the PDF417 code.
    """
    try:
        # Paths to required JAR files
        # These files need to be downloaded and placed in the same directory
        javase_jar = "javase-3.5.0.jar"
        core_jar = "core-3.5.0.jar"
        jcommander_jar = "jcommander-1.82.jar"
        
        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Error: {image_path} not found!")
            return None
            
        # Check for required JAR files
        jar_files = [javase_jar, core_jar, jcommander_jar]
        missing_files = [file for file in jar_files if not os.path.exists(file)]
        
        if missing_files:
            print("Missing required JAR files. Downloading them now...")
            # URLs for downloading the JAR files
            jar_urls = {
                "javase-3.5.0.jar": "https://repo1.maven.org/maven2/com/google/zxing/javase/3.5.0/javase-3.5.0.jar",
                "core-3.5.0.jar": "https://repo1.maven.org/maven2/com/google/zxing/core/3.5.0/core-3.5.0.jar",
                "jcommander-1.82.jar": "https://repo1.maven.org/maven2/com/beust/jcommander/1.82/jcommander-1.82.jar"
            }
            
            # Download missing JAR files
            for file in missing_files:
                try:
                    url = jar_urls.get(file)
                    if url:
                        print(f"Downloading {file} from {url}")
                        subprocess.run(["curl", "-L", "-o", file, url], check=True)
                    else:
                        print(f"URL not found for {file}")
                        return None
                except subprocess.CalledProcessError as e:
                    print(f"Error downloading {file}: {e}")
                    return None
        
        # Preprocess the image to improve detection
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read the image {image_path}")
            return None
            
        # Apply preprocessing techniques
        preprocessed_images = []
        
        # Original image
        preprocessed_images.append(("Original", image_path))
        
        # Grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_path = image_path.replace('.png', '_thresh.png')
        cv2.imwrite(thresh_path, thresh)
        preprocessed_images.append(("Threshold", thresh_path))
        
        # Adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 91, 11)
        adaptive_path = image_path.replace('.png', '_adaptive.png')
        cv2.imwrite(adaptive_path, adaptive_thresh)
        preprocessed_images.append(("Adaptive", adaptive_path))
        
        # Try each preprocessed image with ZXing
        for name, img_path in preprocessed_images:
            print(f"Trying to decode with {name} preprocessing...")
            
            # Docker command to detect the PDF417 code
            docker_command = [
                "docker", "run", "--rm",
                "-v", f"{os.getcwd()}:/app",
                "openjdk:17",
                "java", "-cp",
                f"/app/{javase_jar}:/app/{core_jar}:/app/{jcommander_jar}",
                "com.google.zxing.client.j2se.CommandLineRunner",
                "--try_harder",
                "--pure_barcode",
                "--possibleFormats=PDF_417",
                f"/app/{os.path.basename(img_path)}"
            ]
            
            try:
                # Run the Docker command to get the decoding and position
                result = subprocess.run(docker_command, capture_output=True, text=True, check=True)
                output = result.stdout.strip()
                
                # Check if we got a successful decode
                if "No barcode found" not in output and output:
                    print(f"Decoded PDF417 Code ({name}):")
                    print(output)
                    
                    # Parse the ZXing output for barcode position
                    points = []
                    data = None
                    for line in output.splitlines():
                        if line.startswith("Raw result:"):
                            data = line.split("Raw result:")[1].strip()
                        if line.startswith("  Point"):
                            parts = line.split(":")[1].strip().replace("(", "").replace(")", "").split(",")
                            points.append((int(float(parts[0])), int(float(parts[1]))))
                    
                    # If points are found, draw a bounding polygon
                    if len(points) >= 3:
                        # Load the original image with OpenCV
                        original_image = cv2.imread(image_path)
                        
                        # Draw a polygon connecting the points
                        points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(original_image, [points_array], isClosed=True, color=(0, 255, 0), thickness=2)
                        
                        # Save the annotated image
                        annotated_image_path = image_path.replace('.png', f'_detected_{name}.png')
                        cv2.imwrite(annotated_image_path, original_image)
                        
                        # Display the image with matplotlib
                        plt.figure(figsize=(10, 8))
                        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                        plt.title(f"Detected PDF417 Code ({name})")
                        plt.savefig(annotated_image_path)
                        plt.close()
                        
                        print(f"Annotated image saved as {annotated_image_path}")
                    
                    # Clean up temporary files
                    if name != "Original":
                        os.remove(img_path)
                        
                    return data
            except subprocess.CalledProcessError as e:
                print(f"Error during decoding with {name}:")
                print(e.stderr)
                continue
        
        # Clean up temporary files
        for name, img_path in preprocessed_images:
            if name != "Original" and os.path.exists(img_path):
                os.remove(img_path)
                
        print("No PDF417 code found in the image after trying multiple preprocessing techniques.")
        return None
            
    except Exception as e:
        print(f"Error decoding PDF417 code: {e}")
        return None

if __name__ == "__main__":
    # Path to the image containing the PDF417 code
    image_path = "image.png"
    
    # Decode the PDF417 code
    decoded_data = decode_pdf417_zxing(image_path)
    
    if decoded_data:
        print(f"Decoded data: {decoded_data}")
    else:
        print("Failed to decode PDF417 code.")
        print("Note: This decoder requires Docker to be installed and running.")
        print("Required JAR files will be downloaded automatically if not present.")