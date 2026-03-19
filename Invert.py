import os
from PIL import Image, ImageOps

folder_path = r"C:\Users\aprko\Desktop\Group_Project\drive-download-20260316T155545Z-3-001"

output_folder = os.path.join(folder_path, "grayscale_inverted")
os.makedirs(output_folder, exist_ok=True)

# process APC_0001 to APC_0032
for i in range(1, 17):

    filename = f"APC_{i:04d}.jpg"
    filepath = os.path.join(folder_path, filename)

    if os.path.exists(filepath):

        img = Image.open(filepath)

        # convert to grayscale
        gray = img.convert("L")

        # invert brightness
        inverted = ImageOps.invert(gray)

        # save result
        save_path = os.path.join(output_folder, filename)
        inverted.save(save_path)

        print(f"Processed {filename}")

    else:
        print(f"Missing: {filename}")

print("Done.")