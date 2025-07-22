import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load data
df = pd.read_csv("agglomeration.csv")

# Calculate percentage error and round to integer
df["Error (%)"] = (
    (100 * (df["Auto"] - df["Manual"]).abs() / df["Manual"]).round().astype(int)
)

# Add sequential numbering
df["Sample_ID"] = range(1, len(df) + 1)

# Create figure with two subplots (plot + table)
fig = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 2, width_ratios=[1, 1])
plt.subplots_adjust(wspace=0.3)

# 1. Scatter plot
ax1 = plt.subplot(gs[0])
sc = ax1.scatter(
    df["Agg. Rate"],
    df["Error (%)"],
    c=df["Manual"] / 1e6,
    cmap="viridis",
    s=100,
    alpha=0.8,
)

# Add regression line
z = np.polyfit(df["Agg. Rate"], df["Error (%)"], 1)
p = np.poly1d(z)
ax1.plot(df["Agg. Rate"], p(df["Agg. Rate"]), "r")

# Format plot
ax1.set_xlabel("Agglomeration Rate (%)", fontsize=12)
ax1.set_ylabel("Percentage Error (%)", fontsize=12)
ax1.set_title("Measurement Error vs. Agglomeration Rate", fontsize=14)
ax1.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(sc, ax=ax1)
cbar.set_label("Manual Concentration (×10⁶ cells/mL)", fontsize=10)

# Annotate with sequential numbers
for i, row in df.iterrows():
    ax1.text(
        row["Agg. Rate"] + 2,
        row["Error (%)"],
        str(row["Sample_ID"]),
        fontsize=12,
        ha="left",
        va="center",
        weight="bold",
    )

# 2. Table
ax2 = plt.subplot(gs[1])
ax2.axis("off")

# Prepare table data
table_data = df[["Sample_ID", "Agg. Rate", "Error (%)"]].copy()
table_data["Agg. Rate"] = table_data["Agg. Rate"].astype(int)  # Convert to integer
table_data["Error (%)"] = table_data["Error (%)"].astype(
    int
)  # Already rounded, ensure int
table_data["Manual"] = (df["Manual"] / 1e6).round(2)
table_data["Auto"] = (df["Auto"] / 1e6).round(2)
table_data = table_data.rename(columns={"Sample_ID": "Sample"})
table_data = table_data.sort_values("Sample")

# Create table with integer formatting
cell_text = []
for row in table_data.values:
    cell_text.append(
        [
            str(int(row[0])),  # ID
            str(int(row[1])),  # Agg. Rate (%)
            str(int(row[2])),  # Error (%)
            f"{row[3]:.2f}",  # Manual (1 decimal)
            f"{row[4]:.2f}",
        ]
    )

table = ax2.table(
    cellText=cell_text,
    colLabels=["Sample", "Agg (%)", "Error (%)", "Manual", "Auto"],
    loc="center",
    cellLoc="center",
    colColours=["#f0f0f0"] * 5,
)


# Style table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.5)

# Highlight header
for (i, j), cell in table.get_celld().items():
    if i == 0:  # Header row
        cell.set_facecolor("#404040")
        cell.set_text_props(color="white", weight="bold")

plt.tight_layout()
plt.savefig("agglomeration_analysis.png", dpi=300, bbox_inches="tight")
plt.show()
