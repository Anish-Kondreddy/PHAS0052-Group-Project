"""
Grayscale Conversion and Brightness Inversion for Image Sequence
Date: 19 March 2026

Generative AI was used only to assist with improving docstrings, comments,
and logbook-style documentation. The underlying code logic and implementation
remained the student’s own work.

Brief Description
-----------------
Converts a sequence of images to grayscale, inverts pixel brightness, and
saves the processed outputs to a dedicated folder.
"""

import os
from PIL import Image, ImageOps


# =========================================================
# INPUT FOLDER
# =========================================================

folder_path = r"C:\Users\aprko\Desktop\Group_Project\drive-download-20260316T155545Z-3-001"

# Output directory for processed images
output_folder = os.path.join(folder_path, "grayscale_inverted")
os.makedirs(output_folder, exist_ok=True)


# =========================================================
# IMAGE PROCESSING LOOP
# =========================================================

# Process APC_0001 to APC_0016
for i in range(1, 17):

    filename = f"APC_{i:04d}.jpg"
    filepath = os.path.join(folder_path, filename)

    if os.path.exists(filepath):

        # Load image
        img = Image.open(filepath)

        # Convert to grayscale
        gray = img.convert("L")

        # Invert grayscale brightness
        inverted = ImageOps.invert(gray)

        # Save processed image
        save_path = os.path.join(output_folder, filename)
        inverted.save(save_path)

        print(f"Processed {filename}")

    else:
        print(f"Missing: {filename}")

print("Done.")
