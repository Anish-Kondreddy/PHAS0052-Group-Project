import os
import numpy as np
from PIL import Image

folder_path = r"C:\Users\aprko\Desktop\Group_Project\images"

image_averages = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".jpeg")):
        img_path = os.path.join(folder_path, filename)
        
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        avg_pixel_value = img_array.mean()
        image_averages.append(avg_pixel_value)

        print(f"{filename}: average pixel value = {avg_pixel_value:.2f}")

overall_average = np.mean(image_averages)