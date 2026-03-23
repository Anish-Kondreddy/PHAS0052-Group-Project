import os
import csv
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

# =========================================================
# SETTINGS
# =========================================================
folder = r"C:\Users\aprko\Desktop\Group_Project\PH 12 Images"

start_index = 18
end_index = 32
n_runs = 10
max_workers = 15
save_outputs = False   # False for fair benchmarking, True if you also want saved images

# HSV thresholds for blue region detection
lower_blue = np.array([95, 50, 50])
upper_blue = np.array([145, 255, 255])

# Reduce interference from OpenCV internal multithreading
cv2.setNumThreads(1)

# Output folders
seq_folder = os.path.join(folder, "Sequential_Output")
par_folder = os.path.join(folder, "Parallel_Output")
os.makedirs(seq_folder, exist_ok=True)
os.makedirs(par_folder, exist_ok=True)

# Results file
csv_path = os.path.join(folder, "timing_results.csv")

# Figure save paths
summary_table_path = os.path.join(folder, "runtime_summary_table.png")
bar_chart_path = os.path.join(folder, "runtime_comparison_with_se.png")
line_plot_path = os.path.join(folder, "runtime_per_run.png")

# =========================================================
# IMAGE PROCESSING FUNCTION
# =========================================================
def process_image(i, output_folder=None, save_output=False):
    """
    Process one image by isolating blue regions and reducing noise.

    Parameters
    ----------
    i : int
        Image index.
    output_folder : str or None
        Folder for saving processed image.
    save_output : bool
        Whether to save processed output.

    Returns
    -------
    tuple
        (filename, success, message)
    """
    filename = f"APC_{i:04d}.jpg"
    path = os.path.join(folder, filename)

    img = cv2.imread(path)
    if img is None:
        return filename, False, "not found"

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold blue region
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Noise reduction
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Apply mask
    result = cv2.bitwise_and(img, img, mask=mask)

    # Save only if requested
    if save_output and output_folder is not None:
        output_name = f"APC_{i:04d}_NoiseReduced.jpg"
        output_path = os.path.join(output_folder, output_name)
        cv2.imwrite(output_path, result)
        return filename, True, output_name

    return filename, True, "processed"

# =========================================================
# SEQUENTIAL VERSION
# =========================================================
def run_sequential(save_output=False):
    """
    Run image processing sequentially and return elapsed time.
    """
    t0 = time.perf_counter()

    for i in range(start_index, end_index + 1):
        process_image(i, seq_folder, save_output)

    t1 = time.perf_counter()
    return t1 - t0

# =========================================================
# PARALLEL VERSION
# =========================================================
def run_parallel(save_output=False):
    """
    Run image processing in parallel and return elapsed time.
    """
    t0 = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_image, i, par_folder, save_output)
            for i in range(start_index, end_index + 1)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    t1 = time.perf_counter()
    return t1 - t0

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    sequential_times = []
    parallel_times = []

    print("Running warm-up tests...")
    run_sequential(save_output=False)
    run_parallel(save_output=False)

    print("\nRunning repeated timing tests...\n")

    for run in range(1, n_runs + 1):
        seq_time = run_sequential(save_output=save_outputs)
        par_time = run_parallel(save_output=save_outputs)

        sequential_times.append(seq_time)
        parallel_times.append(par_time)

        print(
            f"Run {run:02d}: "
            f"Sequential = {seq_time:.6f} s, "
            f"Parallel = {par_time:.6f} s"
        )

    # =====================================================
    # CALCULATIONS
    # =====================================================
    seq_mean = np.mean(sequential_times)
    seq_std = np.std(sequential_times, ddof=1) if len(sequential_times) > 1 else 0.0
    seq_se = seq_std / np.sqrt(len(sequential_times)) if len(sequential_times) > 1 else 0.0

    par_mean = np.mean(parallel_times)
    par_std = np.std(parallel_times, ddof=1) if len(parallel_times) > 1 else 0.0
    par_se = par_std / np.sqrt(len(parallel_times)) if len(parallel_times) > 1 else 0.0

    speedup_mean = seq_mean / par_mean if par_mean > 0 else np.inf
    percent_improvement = ((seq_mean - par_mean) / seq_mean) * 100 if seq_mean > 0 else 0.0

    print("\n================ FINAL SUMMARY ================")
    print(f"Sequential mean time: {seq_mean:.6f} s")
    print(f"Sequential std dev:   {seq_std:.6f} s")
    print(f"Sequential std error: {seq_se:.6f} s")
    print(f"Parallel mean time:   {par_mean:.6f} s")
    print(f"Parallel std dev:     {par_std:.6f} s")
    print(f"Parallel std error:   {par_se:.6f} s")
    print(f"Mean speedup:         {speedup_mean:.2f}x")

    if percent_improvement >= 0:
        print(f"Mean improvement:     {percent_improvement:.2f}%")
    else:
        print(f"Parallel was slower by {-percent_improvement:.2f}%")

    # =====================================================
    # SAVE TO CSV
    # =====================================================
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Run", "Sequential Time (s)", "Parallel Time (s)"])
        for i in range(n_runs):
            writer.writerow([i + 1, sequential_times[i], parallel_times[i]])

        writer.writerow([])
        writer.writerow(["Statistic", "Sequential", "Parallel"])
        writer.writerow(["Mean (s)", seq_mean, par_mean])
        writer.writerow(["Standard Deviation (s)", seq_std, par_std])
        writer.writerow(["Standard Error (s)", seq_se, par_se])
        writer.writerow(["Speedup (Sequential / Parallel)", speedup_mean, ""])

        if percent_improvement >= 0:
            writer.writerow(["Improvement (%)", percent_improvement, ""])
        else:
            writer.writerow(["Parallel slower by (%)", -percent_improvement, ""])

    print(f"\nTiming results saved to:\n{csv_path}")

    # =====================================================
    # SUMMARY TABLE FIGURE
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    if percent_improvement >= 0:
        performance_note = f"Parallel faster by {percent_improvement:.1f}%"
    else:
        performance_note = f"Parallel slower by {-percent_improvement:.1f}%"

    table_data = [
        ["Sequential", f"{seq_mean:.6f}", f"{seq_std:.6f}", f"{seq_se:.6f}"],
        ["Parallel",   f"{par_mean:.6f}", f"{par_std:.6f}", f"{par_se:.6f}"]
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=["Method", "Mean Time (s)", "Standard Deviation (s)", "Standard Error (s)"],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    plt.title(f"Summary of Runtime Measurements\n{performance_note}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(summary_table_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # =====================================================
    # ERROR BAR PLOT USING STANDARD ERROR
    # =====================================================
    labels = ["Sequential", "Parallel"]
    means = [seq_mean, par_mean]
    ses = [seq_se, par_se]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, means, yerr=ses, capsize=8)

    ax.set_ylabel("Mean Execution Time (seconds)", fontsize=12)
    ax.set_xlabel("Type of Processing", fontsize=12)
    ax.set_title("Mean Runtime Comparison with Standard Error Bars", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    y_offset = max(ses) * 1.5 if max(ses) > 0 else max(means) * 0.03

    for bar, mean_val, se_val in zip(bars, means, ses):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + se_val + y_offset,
            f"{mean_val:.4f} ± {se_val:.4f} s",
            ha="center",
            va="bottom",
            fontsize=11
        )

    summary_text = (
        f"Speedup: {speedup_mean:.2f}x\n"
        f"{performance_note}"
    )

    ax.text(
        0.95, 0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.tight_layout()
    plt.savefig(bar_chart_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # =====================================================
    # PER-RUN LINE PLOT
    # =====================================================
    run_numbers = np.arange(1, n_runs + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(run_numbers, sequential_times, marker="o", label="Sequential")
    ax.plot(run_numbers, parallel_times, marker="o", label="Parallel")

    ax.set_xlabel("Run Number", fontsize=12)
    ax.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax.set_title("Execution Time for Each Repeated Run", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(line_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print("\nSaved figures:")
    print(summary_table_path)
    print(bar_chart_path)
    print(line_plot_path)