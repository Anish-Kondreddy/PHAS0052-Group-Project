import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')  # clean professional base

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 300,
})

# --- SETTINGS ---
lower_blue = np.array([100, 140, 120])
upper_blue = np.array([140, 255, 255])

def process_folder(folder, image_numbers):
    brightness_values = []
    frames = []

    for frame_idx, img_num in enumerate(image_numbers, start=1):
        filename = f"APC_{img_num:04d}.jpg"
        path = os.path.join(folder, filename)

        img = cv2.imread(path)
        if img is None:
            print(f"Skipping {filename} (not found)")
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Blue mask
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Apply mask
        result = cv2.bitwise_and(img, img, mask=mask)

        # Convert to grayscale brightness
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Keep only bright pixels
        _, bright_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Active pixels only
        active_pixels = gray[bright_mask > 0]

        if len(active_pixels) > 0:
            avg_brightness = np.mean(active_pixels)
            brightness_values.append(avg_brightness)
            frames.append(frame_idx)

    return frames, brightness_values


# --- DATASET PATHS ---
folder_ph13 = r"C:\Users\aprko\Desktop\Group_Project\drive-download-20260316T155545Z-3-001"
folder_ph12 = r"C:\Users\aprko\Desktop\Group_Project\drive-download-20260316T162457Z-3-001"

# --- IMAGE NUMBERS ---
# pH 13: APC_0001 to APC_0016  -> frames 1 to 16
# pH 12: APC_0018 to APC_0032  -> frames 1 to 15
frames_13, brightness_13 = process_folder(folder_ph13, range(1, 17))
frames_12, brightness_12 = process_folder(folder_ph12, range(18, 33))

plt.figure(figsize=(8, 5))

# Main lines
plt.plot(frames_13, brightness_13, marker='o', linewidth=2, label='pH 13')
plt.plot(frames_12, brightness_12, marker='s', linewidth=2,
         linestyle='-', color='orange', label='pH 12')

# --- AVERAGES ---
avg_13 = np.mean(brightness_13)
avg_12 = np.mean(brightness_12)

# Mean lines
plt.axhline(avg_13, linestyle=':', linewidth=2,
            label=f'pH 13 mean = {avg_13:.1f}')

plt.axhline(avg_12, linestyle=':', linewidth=2,
            color='orange',
            label=f'pH 12 mean = {avg_12:.1f}')

# Labels
plt.xlabel("Frame Number")
plt.ylabel("Average Brightness (a.u.)")
plt.title("Chemiluminescent Brightness vs Frame")

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig("brightness_vs_frame_with_means.png", dpi=300)
plt.show()