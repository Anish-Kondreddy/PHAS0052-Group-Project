import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# SETTINGS
# =========================
folder_path = r"C:\Users\aprko\Desktop\Group_Project\drive-download-20260316T162457Z-3-001"
output_folder = os.path.join(folder_path, "background_removed_frames")
os.makedirs(output_folder, exist_ok=True)

file_names = [f"APC_{i:04d}.jpg" for i in range(18, 33)]

# threshold for detecting bright activity after background subtraction
threshold = 20  # adjust if needed

# =========================
# LOAD IMAGES
# =========================
images = []
valid_files = []

for file in file_names:
    img_path = os.path.join(folder_path, file)
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("L")   # grayscale
        img_array = np.array(img, dtype=np.float32)
        images.append(img_array)
        valid_files.append(file)
    else:
        print(f"Missing file: {file}")

images = np.array(images)  # shape = (n_frames, height, width)

if len(images) == 0:
    raise ValueError("No images found.")

n_frames = images.shape[0]

# =========================
# MEAN IMAGE
# =========================
mean_image = np.mean(images, axis=0)

# save mean image
mean_img_uint8 = np.clip(mean_image, 0, 255).astype(np.uint8)
Image.fromarray(mean_img_uint8).save(os.path.join(folder_path, "mean_image.jpg"))

# =========================
# BACKGROUND SUBTRACTION
# =========================
# signed difference: keeps whether a pixel is brighter/darker than average
diff_images = images - mean_image

# absolute difference: good for activity detection
abs_diff_images = np.abs(diff_images)

# =========================
# SAVE BACKGROUND-REMOVED FRAMES
# =========================
for i, diff_img in enumerate(diff_images):
    # shift to visible range for saving
    # zero difference becomes mid-grey
    vis = diff_img + 128
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    Image.fromarray(vis).save(
        os.path.join(output_folder, f"bg_removed_{i+1:04d}.jpg")
    )

# Also save absolute-difference images
abs_output_folder = os.path.join(folder_path, "absolute_difference_frames")
os.makedirs(abs_output_folder, exist_ok=True)

for i, abs_img in enumerate(abs_diff_images):
    vis = np.clip(abs_img, 0, 255).astype(np.uint8)
    Image.fromarray(vis).save(
        os.path.join(abs_output_folder, f"abs_diff_{i+1:04d}.jpg")
    )

# =========================
# METRICS PER FRAME
# =========================
frame_numbers = np.arange(1, n_frames + 1)

mean_brightness = np.mean(images, axis=(1, 2))
max_brightness = np.max(images, axis=(1, 2))
std_brightness = np.std(images, axis=(1, 2))

mean_abs_diff = np.mean(abs_diff_images, axis=(1, 2))
max_abs_diff = np.max(abs_diff_images, axis=(1, 2))
std_abs_diff = np.std(abs_diff_images, axis=(1, 2))

# number of pixels above threshold after background subtraction
bright_pixel_count = np.sum(abs_diff_images > threshold, axis=(1, 2))

# total intensity of active pixels above threshold
active_intensity_sum = np.sum(np.where(abs_diff_images > threshold, abs_diff_images, 0), axis=(1, 2))

# =========================
# PLOTS
# =========================

# 1. Mean brightness of original frames
plt.figure(figsize=(8, 5))
plt.plot(frame_numbers, mean_brightness, marker='o')
plt.xlabel("Frame number")
plt.ylabel("Mean pixel brightness")
plt.title("Mean Brightness of Original Frames")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Max brightness of original frames
plt.figure(figsize=(8, 5))
plt.plot(frame_numbers, max_brightness, marker='o')
plt.xlabel("Frame number")
plt.ylabel("Max pixel brightness")
plt.title("Max Brightness of Original Frames")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Standard deviation of original frames
plt.figure(figsize=(8, 5))
plt.plot(frame_numbers, std_brightness, marker='o')
plt.xlabel("Frame number")
plt.ylabel("Standard deviation of pixel brightness")
plt.title("Brightness Standard Deviation of Original Frames")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Mean absolute difference from mean image
plt.figure(figsize=(8, 5))
plt.plot(frame_numbers, mean_abs_diff, marker='o')
plt.xlabel("Frame number")
plt.ylabel("Mean absolute difference")
plt.title("Mean Absolute Difference from Mean Image")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Standard deviation of absolute difference
plt.figure(figsize=(8, 5))
plt.plot(frame_numbers, std_abs_diff, marker='o')
plt.xlabel("Frame number")
plt.ylabel("Std. dev. of absolute difference")
plt.title("Variation After Background Subtraction")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Number of bright active pixels
plt.figure(figsize=(8, 5))
plt.plot(frame_numbers, bright_pixel_count, marker='o')
plt.xlabel("Frame number")
plt.ylabel(f"Pixels with abs(diff) > {threshold}")
plt.title("Number of Active Bright Pixels")
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Total active intensity above threshold
plt.figure(figsize=(8, 5))
plt.plot(frame_numbers, active_intensity_sum, marker='o')
plt.xlabel("Frame number")
plt.ylabel("Sum of active pixel intensity")
plt.title("Total Bright Activity After Background Subtraction")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# SHOW MEAN IMAGE
# =========================
plt.figure(figsize=(6, 6))
plt.imshow(mean_image, cmap="gray")
plt.title("Mean Image")
plt.axis("off")
plt.tight_layout()
plt.show()

# =========================
# OPTIONAL: show one example frame and its cleaned version
# =========================
example_index = 0  # first frame

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(images[example_index], cmap="gray")
plt.title(f"Original: {valid_files[example_index]}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(diff_images[example_index], cmap="gray")
plt.title("Signed Difference")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(abs_diff_images[example_index], cmap="gray")
plt.title("Absolute Difference")
plt.axis("off")

plt.tight_layout()
plt.show()