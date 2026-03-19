import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

# Path to the image
img_path = r"C:\Users\aprko\Desktop\Group_Project\drive-download-20260316T162457Z-3-001\APC_0018.jpg"

# Load image in grayscale
img = Image.open(img_path).convert("L")
img_array = np.array(img, dtype=np.float32)

# Remove slow background variations
background = gaussian_filter(img_array, sigma=20)
img_highpass = img_array - background

# Apply Hanning window (reduces FFT edge effects)
h, w = img_highpass.shape
window_y = np.hanning(h)
window_x = np.hanning(w)
window = np.outer(window_y, window_x)
img_processed = img_highpass * window

# 2D FFT
fft2 = np.fft.fft2(img_processed)
fft_shift = np.fft.fftshift(fft2)
power_spectrum = np.abs(fft_shift) ** 2

# Remove central DC component
cy, cx = h // 2, w // 2
power_spectrum[cy-5:cy+6, cx-5:cx+6] = 0

# Radial averaging
y, x = np.indices((h, w))
r = np.sqrt((x - cx)**2 + (y - cy)**2)
r_int = r.astype(int)

radial_sum = np.bincount(r_int.ravel(), weights=power_spectrum.ravel())
radial_count = np.bincount(r_int.ravel())
radial_profile = radial_sum / radial_count

# Ignore very low frequencies
min_radius = 10
peak_radius = np.argmax(radial_profile[min_radius:]) + min_radius

# Spatial frequency and wavelength
freq_per_pixel = peak_radius / max(h, w)
wavelength_pixels = 1 / freq_per_pixel if freq_per_pixel > 0 else np.inf

print(f"Dominant spatial frequency: {freq_per_pixel:.4f} cycles/pixel")
print(f"Estimated spatial wavelength: {wavelength_pixels:.2f} pixels")

# Plot original image
plt.figure()
plt.imshow(img_array, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Plot background removed image
plt.figure()
plt.imshow(img_highpass, cmap="gray")
plt.title("Background Removed")
plt.axis("off")

# Plot power spectrum
plt.figure()
plt.imshow(np.log1p(power_spectrum), cmap="gray")
plt.title("Log Power Spectrum")
plt.axis("off")

# Plot radial spectrum
plt.figure()
plt.plot(radial_profile)
plt.axvline(peak_radius, linestyle="--", label="Peak")
plt.xlabel("Radius (FFT pixels)")
plt.ylabel("Power")
plt.title("Radial Power Spectrum")
plt.legend()
plt.show()