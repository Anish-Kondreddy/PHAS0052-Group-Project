"""
Average Pixel Brightness Across Image Folder
Date: 19 March 2026

Generative AI was used only to assist with improving docstrings, comments,
and logbook-style documentation. The underlying code logic and implementation
remained the student’s own work.

Brief Description
-----------------
Computes the average pixel brightness for each image in a folder and
calculates the overall mean brightness across all images.
"""

import os
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------
# IMAGE FOLDER PATH
# ---------------------------------------------------------------------

folder_path = r"C:\Users\aprko\Desktop\Group_Project\images"


# ---------------------------------------------------------------------
# COMPUTE PER-IMAGE AVERAGE PIXEL VALUES
# ---------------------------------------------------------------------

image_averages = []

for filename in os.listdir(folder_path):

    # Process only JPEG images
    if filename.lower().endswith((".jpg", ".jpeg")):

        img_path = os.path.join(folder_path, filename)

        # Load image and convert to RGB format
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        # Compute mean pixel intensity across all channels
        avg_pixel_value = img_array.mean()

        image_averages.append(avg_pixel_value)

        print(f"{filename}: average pixel value = {avg_pixel_value:.2f}")


# ---------------------------------------------------------------------
# COMPUTE OVERALL DATASET AVERAGE
# ---------------------------------------------------------------------

overall_average = np.mean(image_averages)
