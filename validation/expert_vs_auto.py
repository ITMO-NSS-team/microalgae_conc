import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def form_autosegmented_concentrations():
    """
    No valid docstring found.
    """

    images_path = "new_data"

    auto_segmented_df = pd.DataFrame()
    names = []

    for s in range(1, 17):
        if s != 11:
            names.append(s)
            file = f"{images_path}/{s}/segmented/cells.csv"
            df = pd.read_csv(file)
            auto_segmented_df[s] = df["concentration"]
    auto_segmented_df = auto_segmented_df.T
    auto_segmented_df.columns = [
        "im1",
        "im2",
        "im3",
        "im4",
        "im5",
        "im6",
        "im7",
        "im8",
        "im9",
        "im10",
    ]
    auto_segmented_df["Sample"] = names
    auto_segmented_df = auto_segmented_df[
        [
            "Sample",
            "im1",
            "im2",
            "im3",
            "im4",
            "im5",
            "im6",
            "im7",
            "im8",
            "im9",
            "im10",
        ]
    ]
    auto_segmented_df.to_csv("auto_concentration.csv", index=False)


path_to_expert = "expert_concentration.csv"
path_to_auto = "auto_concentration.csv"

expert_df = pd.read_csv(path_to_expert)
auto_df = pd.read_csv(path_to_auto)

# Prepare data - ensure both datasets have same samples in same order
expert_data = expert_df.drop(columns=["Sample", "Concentration"]).to_numpy().T
auto_data = auto_df.drop(columns=["Sample"]).to_numpy().T

# Create positions for box pairs
n_samples = len(auto_df["Sample"])
positions = np.arange(n_samples) * 1.5  # 1.5 units apart for each sample pair

plt.rcParams["figure.figsize"] = (9, 4)  # Wider figure for better spacing

# Create boxplots
box1 = plt.boxplot(
    expert_data,
    positions=positions - 0.2,  # Offset left
    widths=0.4,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue"),
    medianprops=dict(color="blue"),
    labels=auto_df["Sample"],
)

box2 = plt.boxplot(
    auto_data,
    positions=positions + 0.2,  # Offset right
    widths=0.4,
    patch_artist=True,
    boxprops=dict(facecolor="lightgreen"),
    medianprops=dict(color="green"),
)
plt.title("Comparison of manually and automatic cell concentration estimation ")
plt.ylabel("Cells/ml")
plt.xlabel("Sample number")
plt.xticks(positions, auto_df["Sample"])  # Center x-ticks between pairs
plt.yscale("log")
plt.legend(
    [box1["boxes"][0], box2["boxes"][0]], ["Manual", "Automatic"], loc="lower right"
)

plt.grid(which="minor", alpha=0.2)
plt.grid(which="major", alpha=0.5)
plt.tight_layout()
# plt.savefig('comparison_boxplot.png', dpi=400, bbox_inches='tight')
plt.show()


# ____________________________#
#          Plot with sorted              #
# ____________________________#
# Calculate mean concentrations for sorting
mean_values = np.mean(np.concatenate([expert_data, auto_data]), axis=0)
sort_order = np.argsort(mean_values)

# Sort all data by mean concentration
sorted_samples = np.array(auto_df["Sample"])[sort_order]
sorted_samples = [f"No.{s}" for s in sorted_samples]
expert_data_sorted = expert_data[:, sort_order]
auto_data_sorted = auto_data[:, sort_order]

# Create positions for box pairs with wider spacing
n_samples = len(sorted_samples)
positions = np.arange(n_samples) * 1.5  # Increased from 1.5 to 2.0 for wider spacing

plt.rcParams["figure.figsize"] = (9, 4)  # Wider figure for better spacing

# Create boxplots
box1 = plt.boxplot(
    expert_data_sorted,
    positions=positions - 0.2,  # Offset left
    widths=0.4,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue"),
    medianprops=dict(color="blue"),
    labels=auto_df["Sample"],
)

box2 = plt.boxplot(
    auto_data_sorted,
    positions=positions + 0.2,  # Offset right
    widths=0.4,
    patch_artist=True,
    boxprops=dict(facecolor="lightgreen"),
    medianprops=dict(color="green"),
)
plt.title("Comparison of manual and automatic cell concentration estimation ")
plt.ylabel("Cells/ml")
plt.xlabel("Sample number (sorted by mean concentration)")
plt.xticks(positions, sorted_samples)  # Center x-ticks between pairs
plt.yscale("log")
plt.legend(
    [box1["boxes"][0], box2["boxes"][0]], ["Manual", "Automatic"], loc="lower right"
)
plt.grid(which="minor", alpha=0.2)
plt.grid(which="major", alpha=0.5)
plt.tight_layout()
plt.savefig("sorted_comparison_boxplot.png", dpi=400, bbox_inches="tight")
plt.show()
