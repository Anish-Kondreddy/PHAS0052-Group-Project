"""
Sequential and Parallel Image Processing Comparison
Date: 19 March 2026

Generative AI was used only to assist with improving docstrings, comments,
and logbook-style documentation. The underlying code logic and implementation
remained the student’s own work.

Brief Description
-----------------
Processes a selected image folder using both sequential and parallel methods,
then compares execution times and displays a timing plot.
"""

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures


# =========================================================
# SETTINGS
# =========================================================

ph13 = r"C:\Users\aprko\Desktop\Group_Project\PH 13 Images"
ph12 = r"C:\Users\aprko\Desktop\Group_Project\PH 12 Images"

# Choose which folder to process
folder = ph13   # Change to ph12 if needed

start_index = 18
end_index = 32

# HSV thresholds for blue detection
lower_blue = np.array([95, 50, 50])
upper_blue = np.array([145, 255, 255])

# Output folders
seq_folder = os.path.join(folder, "Sequential_Output")
par_folder = os.path.join(folder, "Parallel_Output")
os.makedirs(seq_folder, exist_ok=True)
os.makedirs(par_folder, exist_ok=True)


# =========================================================
# IMAGE PROCESSING FUNCTION
# =========================================================

def process_image(i, input_folder, output_folder):
    """Process a single image and save the masked output."""
    filename = f"APC_{i:04d}.jpg"
    path = os.path.join(input_folder, filename)

    img = cv2.imread(path)

    if img is None:
        return filename, False, "not found"

    # Convert to HSV colour space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create blue-region mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply basic noise reduction
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Apply mask to original image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Save processed output
    output_name = f"APC_{i:04d}_NoiseReduced.jpg"
    output_path = os.path.join(output_folder, output_name)
    cv2.imwrite(output_path, result)

    return filename, True, output_name


# =========================================================
# SEQUENTIAL VERSION
# =========================================================

def run_sequential():
    """Run image processing sequentially and measure execution time."""
    t0 = time.perf_counter()
    results = []

    for i in range(start_index, end_index + 1):
        results.append(process_image(i, folder, seq_folder))

    t1 = time.perf_counter()
    elapsed = t1 - t0
    return results, elapsed


# =========================================================
# PARALLEL VERSION
# =========================================================

def run_parallel():
    """Run image processing in parallel and measure execution time."""
    t0 = time.perf_counter()
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_image, i, folder, par_folder)
            for i in range(start_index, end_index + 1)
        ]

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    t1 = time.perf_counter()
    elapsed = t1 - t0
    return results, elapsed


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print(f"Processing folder: {folder}")

    print("\nRunning sequential version...")
    seq_results, seq_time = run_sequential()

    print("\nRunning parallel version...")
    par_results, par_time = run_parallel()

    # Sort results so printed output is ordered by filename
    seq_results.sort(key=lambda x: x[0])
    par_results.sort(key=lambda x: x[0])

    print("\n--- Sequential results ---")
    for filename, success, msg in seq_results:
        if success:
            print(f"{filename} -> Saved: {msg}")
        else:
            print(f"{filename} -> Skipping ({msg})")

    print("\n--- Parallel results ---")
    for filename, success, msg in par_results:
        if success:
            print(f"{filename} -> Saved: {msg}")
        else:
            print(f"{filename} -> Skipping ({msg})")

    # =====================================================
    # TIMING COMPARISON
    # =====================================================

    print("\n================ TIMING COMPARISON ================")
    print(f"Sequential time: {seq_time:.6f} seconds")
    print(f"Parallel time:   {par_time:.6f} seconds")

    if par_time > 0:
        speedup = seq_time / par_time
        print(f"Speedup factor:  {speedup:.2f}x")
    else:
        speedup = float("inf")
        print("Speedup factor: Infinite (parallel time too small to measure)")

    if seq_time > 0:
        percent_quicker = ((seq_time - par_time) / seq_time) * 100
        print(f"Parallel was {percent_quicker:.2f}% quicker")
    else:
        percent_quicker = 0
        print("Could not calculate percentage quicker")

    # =====================================================
    # TIMING PLOT
    # =====================================================

    labels = ["Sequential", "Parallel"]
    times = [seq_time, par_time]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, times)

    plt.ylabel("Execution Time (seconds)", fontsize=12)
    plt.title("Performance Comparison: Sequential vs Parallel Image Processing", fontsize=14)

    # Add exact timing values above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4f} s",
            ha="center",
            va="bottom",
            fontsize=11
        )

    # Add timing summary box
    textstr = f"Speedup: {speedup:.2f}x\nImprovement: {percent_quicker:.1f}%"
    plt.gca().text(
        0.95,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
