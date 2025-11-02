import cv2 
import numpy as np 
import subprocess 
import os 
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect and decode PDF417 barcodes using ZXing')
    parser.add_argument('--image', '-i', default="id-gabriel.png", help='Path to the image containing PDF417 code')
    args = parser.parse_args()
    
    barcode_image = args.image
    
    # Paths to required files 
    javase_jar = "javase-3.5.0.jar" 
    core_jar = "core-3.5.0.jar" 
    jcommander_jar = "jcommander-1.82.jar" 
    
    # Validate required files 
    for file in [javase_jar, core_jar, jcommander_jar, barcode_image]: 
        if not os.path.exists(file): 
            print(f"Error: {file} not found!") 
            exit(1) 
    
    # Docker command to detect the barcode and get its position 
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
        f"/app/{barcode_image}" 
    ] 
    
    try: 
        # Run the Docker command to get the decoding and position 
        result = subprocess.run(docker_command, capture_output=True, text=True, check=True) 
        output = result.stdout.strip() 
        print("Decoded Output:") 
        print(output) 
    except subprocess.CalledProcessError as e: 
        print("Error during decoding:") 
        print(e.stderr) 
        
        # Try with Python libraries as fallback
        print("\nTrying with Python libraries as fallback...")
        try:
            from pyzbar.pyzbar import decode
            
            # Read the image
            image = cv2.imread(barcode_image)
            if image is None:
                print(f"Error: Unable to read the image {barcode_image}")
                exit(1)
                
            # Try to decode with pyzbar
            decoded_objects = decode(image)
            if decoded_objects:
                for obj in decoded_objects:
                    if obj.type == 'PDF417':
                        print(f"Decoded PDF417 with pyzbar:")
                        print(f"Data: {obj.data.decode('utf-8')}")
                        
                        # Draw the barcode location
                        points = obj.polygon
                        if len(points) > 0:
                            # Convert points to numpy array for drawing
                            points_array = np.array(points, np.int32).reshape((-1, 1, 2))
                            
                            # Draw polygon
                            cv2.polylines(image, [points_array], isClosed=True, color=(0, 255, 0), thickness=2)
                            
                            # Save and display the annotated image
                            annotated_image_path = "annotated_pdf417.png" 
                            cv2.imwrite(annotated_image_path, image) 
                            print(f"Annotated image saved as {annotated_image_path}") 
                            
                            # Display the image 
                            cv2.imshow("Detected PDF417", image) 
                            print("Press any key to close the window.") 
                            cv2.waitKey(0) 
                            cv2.destroyAllWindows()
                            exit(0)
            
            print("No PDF417 code found with pyzbar.")
            exit(1)
        except Exception as e:
            print(f"Error with pyzbar fallback: {e}")
            exit(1)
    
    # Parse the ZXing output for barcode position 
    points = [] 
    for line in output.splitlines(): 
        if line.startswith("  Point"): 
            parts = line.split(":")[1].strip().replace("(", "").replace(")", "").split(",") 
            points.append((int(float(parts[0])), int(float(parts[1])))) 
    
    # If points are found, draw a bounding polygon 
    if len(points) >= 4: 
        # Load the image with OpenCV 
        image = cv2.imread(barcode_image) 
        if image is None: 
            print("Error: Unable to read the image!") 
            exit(1) 
    
        # Draw a polygon connecting the points 
        points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2)) 
        print(f"Drawing polygon with points: {points}") 
    
        cv2.polylines(image, [points_array], isClosed=True, color=(0, 255, 0), thickness=2) 
    
        # Save and display the annotated image 
        annotated_image_path = "annotated_pdf417.png" 
        cv2.imwrite(annotated_image_path, image) 
        print(f"Annotated image saved as {annotated_image_path}") 
    
        # Display the image 
        cv2.imshow("Detected PDF417", image) 
        print("Press any key to close the window.") 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
    else: 
        print("No bounding box points detected.") 

if __name__ == "__main__":
    main()