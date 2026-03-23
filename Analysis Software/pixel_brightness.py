"""
Interactive Pixel Brightness Inspector
Date: 19 March 2026

Generative AI was used only to assist with improving docstrings, comments,
and logbook-style documentation. The underlying code logic and implementation
remained the student’s own work.

Brief Description
-----------------
Displays an image and prints pixel brightness values when clicked.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------------------
# IMAGE PATH
# ---------------------------------------------------------------------

img_path = r"C:\Users\aprko\Desktop\Group_Project\images\test2.jpg"


# ---------------------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------------------

img = Image.open(img_path).convert("RGB")
img_array = np.array(img)


# ---------------------------------------------------------------------
# COMPUTE GRAYSCALE BRIGHTNESS
# ---------------------------------------------------------------------

brightness = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])


# ---------------------------------------------------------------------
# DISPLAY IMAGE WITH CLICK INTERACTION
# ---------------------------------------------------------------------

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Click anywhere to see pixel brightness")


def onclick(event):
    """Print brightness value at clicked pixel location."""
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        value = brightness[y, x]
        print(f"Clicked at (x={x}, y={y}) → Brightness: {value:.2f}")


fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()
