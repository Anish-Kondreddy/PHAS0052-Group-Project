"""
FFT-Based Spatial Frequency and Wavelength Estimation
Date: 19 March 2026

Generative AI was used only to assist with improving docstrings, comments,
and logbook-style documentation. The underlying code logic and implementation
remained the student’s own work.

Brief Description
-----------------
Loads a grayscale image, removes slow background variation, applies a window
function, performs a 2D FFT, and estimates the dominant spatial wavelength
from the radial power spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter


# =========================================================
# IMAGE PATH
# =========================================================

img_path = r"C:\Users\aprko\Desktop\Group_Project\drive-download-20260316T162457Z-3-001\APC_0018.jpg"


# =========================================================
# LOAD IMAGE
# =========================================================

img = Image.open(img_path).convert("L")
img_array = np.array(img, dtype=np.float32)


# =========================================================
# BACKGROUND REMOVAL
# =========================================================

# Remove slow spatial background variations using Gaussian smoothing
background = gaussian_filter(img_array, sigma=20)
img_highpass = img_array - background


# =========================================================
# WINDOWING BEFORE FFT
# =========================================================

# Apply a Hanning window to reduce edge effects in the FFT
h, w = img_highpass.shape
window_y = np.hanning(h)
window_x = np.hanning(w)
window = np.outer(window_y, window_x)

img_processed = img_highpass * window


# =========================================================
# 2D FFT AND POWER SPECTRUM
# =========================================================

fft2 = np.fft.fft2(img_processed)
fft_shift = np.fft.fftshift(fft2)
power_spectrum = np.abs(fft_shift) ** 2


# =========================================================
# REMOVE CENTRAL DC COMPONENT
# =========================================================

cy, cx = h // 2, w // 2
power_spectrum[cy - 5:cy + 6, cx - 5:cx + 6] = 0


# =========================================================
# RADIAL AVERAGING
# =========================================================

y, x = np.indices((h, w))
r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
r_int = r.astype(int)

radial_sum = np.bincount(r_int.ravel(), weights=power_spectrum.ravel())
radial_count = np.bincount(r_int.ravel())
radial_profile = radial_sum / radial_count


# =========================================================
# DOMINANT SPATIAL FREQUENCY
# =========================================================

# Ignore very low frequencies when locating the dominant peak
min_radius = 10
peak_radius = np.argmax(radial_profile[min_radius:]) + min_radius

freq_per_pixel = peak_radius / max(h, w)
wavelength_pixels = 1 / freq_per_pixel if freq_per_pixel > 0 else np.inf

print(f"Dominant spatial frequency: {freq_per_pixel:.4f} cycles/pixel")
print(f"Estimated spatial wavelength: {wavelength_pixels:.2f} pixels")


# =========================================================
# PLOTS
# =========================================================

# Original image
plt.figure()
plt.imshow(img_array, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Background-removed image
plt.figure()
plt.imshow(img_highpass, cmap="gray")
plt.title("Background Removed")
plt.axis("off")

# Log power spectrum
plt.figure()
plt.imshow(np.log1p(power_spectrum), cmap="gray")
plt.title("Log Power Spectrum")
plt.axis("off")

# Radial power spectrum
plt.figure()
plt.plot(radial_profile)
plt.axvline(peak_radius, linestyle="--", label="Peak")
plt.xlabel("Radius (FFT pixels)")
plt.ylabel("Power")
plt.title("Radial Power Spectrum")
plt.legend()
plt.show()
