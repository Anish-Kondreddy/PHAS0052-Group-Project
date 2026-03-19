import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

folder_path = r"C:\Users\aprko\Desktop\Group_Project\images"

files = [
    "PH7.jpeg",
    "PH10.jpeg",
    "PH11.jpeg",
    "PH12.jpeg",
    "PH13.jpeg"
]

ph_values = []
max_brightness = []
avg_brightness = []

for file in files:
    
    ph = int(file.replace("PH","").replace(".jpeg",""))
    ph_values.append(ph)

    img_path = os.path.join(folder_path, file)
    
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)

    max_pixel = img_array.max()
    avg_pixel = img_array.mean()

    max_brightness.append(max_pixel)
    avg_brightness.append(avg_pixel)

# calculate ΔpH relative to pH 7
delta_ph = ph_values

# -------- Plot 1: ΔpH vs Max Brightness --------
plt.figure()
plt.scatter(delta_ph, max_brightness)
plt.plot(delta_ph, max_brightness)
plt.xlabel("ΔpH")
plt.ylabel("Max Pixel Brightness")
plt.title("ΔpH vs Max Pixel Brightness")
plt.show()

# -------- Plot 2: ΔpH vs Average Brightness --------
plt.figure()
plt.scatter(delta_ph, avg_brightness)
plt.plot(delta_ph, avg_brightness)
plt.xlabel("ΔpH")
plt.ylabel("Average Pixel Brightness")
plt.title("ΔpH vs Average Pixel Brightness")
plt.show()