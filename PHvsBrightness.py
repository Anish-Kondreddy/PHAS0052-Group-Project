import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# -----------------------------
# Professional plot formatting
# -----------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 300
})

# -----------------------------
# Folder
# -----------------------------
folder = r"C:\Users\aprko\Desktop\Group_Project\images"

# -----------------------------
# Relaxed constraints for weaker spots
# HSV = [Hue, Saturation, Value]
# -----------------------------
lower_blue = np.array([85, 50, 40])
upper_blue = np.array([155, 255, 255])

# Lower grayscale threshold so weaker active spots remain
gray_threshold = 20

# Images to save after processing
save_ph_set = {7, 10, 11, 12, 13}

ph_values = []
brightness_values = []

for ph in range(7, 14):
    filename = f"PH{ph}.jpeg"
    path = os.path.join(folder, filename)

    img = cv2.imread(path)
    if img is None:
        print(f"Skipping {filename} (not found)")
        continue

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Blue mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Very gentle cleanup so weak spots are not removed
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply mask
    result = cv2.bitwise_and(img, img, mask=mask)

    # Convert masked result to grayscale brightness
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Keep weak-to-strong active pixels
    _, bright_mask = cv2.threshold(gray, gray_threshold, 255, cv2.THRESH_BINARY)

    # Final filtered result
    filtered_result = cv2.bitwise_and(result, result, mask=bright_mask)

    # Active pixels only
    active_pixels = gray[bright_mask > 0]

    if len(active_pixels) > 0:
        avg_brightness = np.mean(active_pixels)
    else:
        avg_brightness = 0

    ph_values.append(ph)
    brightness_values.append(avg_brightness)

    print(f"{filename}: active pixels = {len(active_pixels)}, average brightness = {avg_brightness:.2f}")

    # Save selected processed images
    if ph in save_ph_set:
        save_name = f"PH{ph}_Reduced.jpeg"
        save_path = os.path.join(folder, save_name)
        cv2.imwrite(save_path, filtered_result)
        print(f"Saved processed image: {save_name}")

# -----------------------------
# Line plot
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(ph_values, brightness_values, marker='o', linewidth=2)

plt.xlabel("pH")
plt.ylabel("Average Brightness of Active Blue Spots (a.u.)")
plt.title("Average Chemiluminescent Brightness vs pH")
plt.xticks(ph_values)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(folder, "brightness_vs_pH.png"), dpi=300)
plt.show()

# -----------------------------
# Bar chart
# -----------------------------
plt.figure(figsize=(8, 5))
plt.bar(ph_values, brightness_values)

plt.xlabel("pH")
plt.ylabel("Average Brightness of Active Blue Spots (a.u.)")
plt.title("Average Chemiluminescent Brightness vs pH")
plt.xticks(ph_values)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(folder, "brightness_vs_pH_bar.png"), dpi=300)
plt.show()