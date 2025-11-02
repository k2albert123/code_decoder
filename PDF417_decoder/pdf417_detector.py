import cv2
import numpy as np
import subprocess
import os
import argparse
import sys

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect and decode PDF417 barcodes using ZXing')
    parser.add_argument('--image', '-i', required=True, help='Path to the image containing PDF417 code')
    parser.add_argument('--no-display', action='store_true', help='Do not display the image with detection')
    args = parser.parse_args()
    
    # Paths to required files
    javase_jar = "javase-3.5.0.jar"
    core_jar = "core-3.5.0.jar"
    jcommander_jar = "jcommander-1.82.jar"
    
    barcode_image = args.image
    
    # Validate required files
    missing_files = []
    for file in [javase_jar, core_jar, jcommander_jar]:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing JAR files: {', '.join(missing_files)}")
        print("Downloading required JAR files...")
        
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
                    print(f"Downloaded {file}")
                else:
                    print(f"URL not found for {file}")
                    sys.exit(1)
            except subprocess.CalledProcessError as e:
                print(f"Error downloading {file}: {e}")
                sys.exit(1)
    
    if not os.path.exists(barcode_image):
        print(f"Error: Image {barcode_image} not found!")
        sys.exit(1)
    
    # Preprocess the image to improve detection
    print(f"Preprocessing image: {barcode_image}")
    image = cv2.imread(barcode_image)
    if image is None:
        print(f"Error: Unable to read the image {barcode_image}")
        sys.exit(1)
    
    # Create preprocessed versions of the image
    preprocessed_images = []
    
    # Original image
    preprocessed_images.append(("original", barcode_image))
    
    # Grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_path = barcode_image.replace('.png', '_thresh.png')
    if not barcode_image.endswith('.png'):
        thresh_path = barcode_image + '_thresh.png'
    cv2.imwrite(thresh_path, thresh)
    preprocessed_images.append(("threshold", thresh_path))
    
    # Adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 91, 11)
    adaptive_path = barcode_image.replace('.png', '_adaptive.png')
    if not barcode_image.endswith('.png'):
        adaptive_path = barcode_image + '_adaptive.png'
    cv2.imwrite(adaptive_path, adaptive_thresh)
    preprocessed_images.append(("adaptive", adaptive_path))
    
    # Sharpened image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    sharp_path = barcode_image.replace('.png', '_sharp.png')
    if not barcode_image.endswith('.png'):
        sharp_path = barcode_image + '_sharp.png'
    cv2.imwrite(sharp_path, sharpened)
    preprocessed_images.append(("sharpened", sharp_path))
    
    # Try each preprocessed image with ZXing
    success = False
    for name, img_path in preprocessed_images:
        print(f"\nTrying to decode with {name} preprocessing...")
        
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
                print(f"Decoded PDF417 Code with {name} preprocessing:")
                print(output)
                
                # Parse the ZXing output for barcode position
                points = []
                for line in output.splitlines():
                    if line.startswith("  Point"):
                        parts = line.split(":")[1].strip().replace("(", "").replace(")", "").split(",")
                        points.append((int(float(parts[0])), int(float(parts[1]))))
                
                # If points are found, draw a bounding polygon
                if len(points) >= 3:
                    # Load the original image with OpenCV
                    image = cv2.imread(barcode_image)
                    
                    # Draw a polygon connecting the points
                    points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                    print(f"Drawing polygon with points: {points}")
                    
                    cv2.polylines(image, [points_array], isClosed=True, color=(0, 255, 0), thickness=2)
                    
                    # Save the annotated image
                    annotated_image_path = barcode_image.replace('.png', '_annotated.png')
                    if not barcode_image.endswith('.png'):
                        annotated_image_path = barcode_image + '_annotated.png'
                    cv2.imwrite(annotated_image_path, image)
                    print(f"Annotated image saved as {annotated_image_path}")
                    
                    # Display the image
                    if not args.no_display:
                        cv2.imshow("Detected PDF417 Code", image)
                        print("Press any key to close the window.")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    
                    success = True
                    break
                else:
                    print("No bounding box points detected.")
            else:
                print(f"No PDF417 code found with {name} preprocessing.")
        except subprocess.CalledProcessError as e:
            print(f"Error during decoding with {name}:")
            print(e.stderr)
    
    # Clean up temporary files
    for name, img_path in preprocessed_images:
        if name != "original" and os.path.exists(img_path):
            os.remove(img_path)
    
    if not success:
        print("\nFailed to decode PDF417 code after trying multiple preprocessing techniques.")
        print("Tips for better detection:")
        print("1. Ensure the image has good lighting and contrast")
        print("2. Make sure the PDF417 code is clearly visible and not damaged")
        print("3. Try capturing the image from a different angle or with better lighting")
        print("4. Ensure Docker is installed and running")
        sys.exit(1)

if __name__ == "__main__":
    main()