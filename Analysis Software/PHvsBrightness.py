"""
Brightness Extraction and pH Comparison for Processed Images
Date: 19 March 2026

Generative AI was used only to assist with improving docstrings, comments,
and logbook-style documentation. The underlying code logic and implementation
remained the student’s own work.

Brief Description
-----------------
Processes pH-labelled images by isolating blue chemiluminescent regions,
extracting active-pixel brightness values, saving selected processed images,
and plotting brightness as a function of pH.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# PROFESSIONAL PLOT FORMATTING
# =========================================================

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 300
})


# =========================================================
# INPUT FOLDER
# =========================================================

folder = r"C:\Users\aprko\Desktop\Group_Project\images"


# =========================================================
# PROCESSING SETTINGS
# =========================================================

# Relaxed HSV thresholds to retain weaker blue regions
lower_blue = np.array([85, 50, 40])
upper_blue = np.array([155, 255, 255])

# Lower grayscale threshold to retain weaker active spots
gray_threshold = 20

# pH images to save after processing
save_ph_set = {7, 10, 11, 12, 13}


# =========================================================
# STORAGE ARRAYS
# =========================================================

ph_values = []
brightness_values = []


# =========================================================
# IMAGE PROCESSING LOOP
# =========================================================

for ph in range(7, 14):
    filename = f"PH{ph}.jpeg"
    path = os.path.join(folder, filename)

    img = cv2.imread(path)
    if img is None:
        print(f"Skipping {filename} (not found)")
        continue

    # Convert image to HSV colour space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generate blue-region mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply gentle denoising to preserve weaker spots
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Convert masked output to grayscale brightness
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Retain weak-to-strong active pixels
    _, bright_mask = cv2.threshold(gray, gray_threshold, 255, cv2.THRESH_BINARY)

    # Apply final filtered mask
    filtered_result = cv2.bitwise_and(result, result, mask=bright_mask)

    # Extract active pixels only
    active_pixels = gray[bright_mask > 0]

    if len(active_pixels) > 0:
        avg_brightness = np.mean(active_pixels)
    else:
        avg_brightness = 0

    ph_values.append(ph)
    brightness_values.append(avg_brightness)

    print(
        f"{filename}: active pixels = {len(active_pixels)}, "
        f"average brightness = {avg_brightness:.2f}"
    )

    # Save selected processed images
    if ph in save_ph_set:
        save_name = f"PH{ph}_Reduced.jpeg"
        save_path = os.path.join(folder, save_name)
        cv2.imwrite(save_path, filtered_result)
        print(f"Saved processed image: {save_name}")


# =========================================================
# LINE PLOT
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(ph_values, brightness_values, marker="o", linewidth=2)

plt.xlabel("pH")
plt.ylabel("Average Brightness of Active Blue Spots (a.u.)")
plt.title("Average Chemiluminescent Brightness vs pH")
plt.xticks(ph_values)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(folder, "brightness_vs_pH.png"), dpi=300)
plt.show()


# =========================================================
# BAR CHART
# =========================================================

plt.figure(figsize=(8, 5))
plt.bar(ph_values, brightness_values)

plt.xlabel("pH")
plt.ylabel("Average Brightness of Active Blue Spots (a.u.)")
plt.title("Average Chemiluminescent Brightness vs pH")
plt.xticks(ph_values)
plt.grid(True, axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(folder, "brightness_vs_pH_bar.png"), dpi=300)
plt.show()
