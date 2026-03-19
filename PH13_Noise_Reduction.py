import cv2
import numpy as np
import os

# Folder path
folder = r"C:\Users\aprko\Desktop\Group_Project\drive-download-20260316T162457Z-3-001"

# HSV thresholds (tuned for your images)
lower_blue = np.array([95, 50, 50])
upper_blue = np.array([145, 255, 255])

for i in range(18, 33):
    filename = f"APC_{i:04d}.jpg"
    path = os.path.join(folder, filename)
    
    img = cv2.imread(path)
    
    if img is None:
        print(f"Skipping {filename} (not found)")
        continue

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    # Apply mask
    result = cv2.bitwise_and(img, img, mask=mask)

    # Save output
    output_name = f"APC_{i:04d}_NoiseReduced.jpg"
    output_path = os.path.join(folder, output_name)
    
    cv2.imwrite(output_path, result)

    print(f"Saved: {output_name}")