"""
Brightness vs pH Analysis Plotting Script
Date: 19 March 2026

Generative AI was used only to assist with improving docstrings, comments,
and logbook-style documentation. The underlying code logic and implementation
remained the student’s own work.

Brief Description
-----------------
Loads selected images at different pH conditions and plots maximum and
average pixel brightness as a function of pH.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------------------
# IMAGE FOLDER PATH
# ---------------------------------------------------------------------

folder_path = r"C:\Users\aprko\Desktop\Group_Project\images"


# ---------------------------------------------------------------------
# IMAGE FILES USED FOR ANALYSIS
# ---------------------------------------------------------------------

files = [
    "PH7.jpeg",
    "PH10.jpeg",
    "PH11.jpeg",
    "PH12.jpeg",
    "PH13.jpeg"
]


# ---------------------------------------------------------------------
# INITIALISE STORAGE ARRAYS
# ---------------------------------------------------------------------

ph_values = []
max_brightness = []
avg_brightness = []


# ---------------------------------------------------------------------
# EXTRACT BRIGHTNESS METRICS FROM EACH IMAGE
# ---------------------------------------------------------------------

for file in files:

    # Extract pH value from filename
    ph = int(file.replace("PH", "").replace(".jpeg", ""))
    ph_values.append(ph)

    img_path = os.path.join(folder_path, file)

    # Load image and convert to RGB format
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)

    # Compute brightness metrics
    max_pixel = img_array.max()
    avg_pixel = img_array.mean()

    max_brightness.append(max_pixel)
    avg_brightness.append(avg_pixel)


# NOTE:
# These are absolute pH values (not ΔpH relative to pH 7)
x_values = ph_values


# ---------------------------------------------------------------------
# PLOT 1: pH vs MAX PIXEL BRIGHTNESS
# ---------------------------------------------------------------------

plt.figure()
plt.scatter(x_values, max_brightness)
plt.plot(x_values, max_brightness)

plt.xlabel("pH")
plt.ylabel("Max Pixel Brightness")
plt.title("pH vs Max Pixel Brightness")

plt.show()


# ---------------------------------------------------------------------
# PLOT 2: pH vs AVERAGE PIXEL BRIGHTNESS
# ---------------------------------------------------------------------

plt.figure()
plt.scatter(x_values, avg_brightness)
plt.plot(x_values, avg_brightness)

plt.xlabel("pH")
plt.ylabel("Average Pixel Brightness")
plt.title("pH vs Average Pixel Brightness")

plt.show()
