"""
Blue Region Isolation Using HSV Masking
Date: 19 March 2026

Generative AI was used only to assist with improving docstrings, comments,
and logbook-style documentation. The underlying code logic and implementation
remained the student’s own work.

Brief Description
-----------------
Loads an image, applies HSV thresholding to isolate blue regions, and displays
the masked output.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# IMAGE PATH
# =========================================================

img_path = r"C:\Users\aprko\Desktop\Group_Project\drive-download-20260316T155545Z-3-001\APC_0001.jpg"


# =========================================================
# LOAD IMAGE
# =========================================================

img = cv2.imread(img_path)


# =========================================================
# HSV COLOUR SPACE CONVERSION
# =========================================================

# Convert image from OpenCV BGR format to HSV colour space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# =========================================================
# BLUE REGION THRESHOLDING
# =========================================================

lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

# Create binary mask isolating blue regions
mask = cv2.inRange(hsv, lower_blue, upper_blue)


# =========================================================
# APPLY MASK
# =========================================================

result = cv2.bitwise_and(img, img, mask=mask)


# =========================================================
# DISPLAY RESULT
# =========================================================

# Convert BGR to RGB for correct matplotlib display
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

plt.imshow(result_rgb)
plt.title("Blue spots only")
plt.axis("off")
plt.show()
