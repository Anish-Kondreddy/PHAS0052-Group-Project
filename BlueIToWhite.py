import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread(r"C:\Users\aprko\Desktop\Group_Project\drive-download-20260316T155545Z-3-001\APC_0001.jpg")

# Convert BGR (OpenCV default) → HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define blue range (you can tweak this)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

# Create mask (only blue = white, rest = black)
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Apply mask to original image
result = cv2.bitwise_and(img, img, mask=mask)

# Convert for matplotlib (BGR → RGB)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# Show
plt.imshow(result_rgb)
plt.title("Blue spots only")
plt.axis('off')
plt.show()