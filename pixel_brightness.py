import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = r"C:\Users\aprko\Desktop\Group_Project\images\test2.jpg"

img = Image.open(img_path).convert("RGB")
img_array = np.array(img)

# Convert to grayscale brightness
brightness = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Click anywhere to see pixel brightness")

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        value = brightness[y, x]
        print(f"Clicked at (x={x}, y={y}) → Brightness: {value:.2f}")

fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()