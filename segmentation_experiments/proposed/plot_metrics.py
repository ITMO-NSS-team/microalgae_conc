import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("metrics.csv")
df = df.sort_values(["cells_num"])
df = df[df["cells_num"] < 100]

# MAE METRIC
values = []
labels = []
bins = np.arange(20, 50, 5)
for i in range(len(bins[:-1])):
    ts = df[(bins[i] <= df["cells_num"]) & (df["cells_num"] < bins[i + 1])][
        "MAE"
    ].values
    labels.append(f"{bins[i]}-{bins[i+1]}")
    values.append(ts)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].scatter(df["cells_num"], df["MAE"])
axs[0].set_ylabel("MAE")
axs[0].set_xlabel("Cells number")
axs[0].set_title("Each point is image")

axs[1].boxplot(values, tick_labels=labels)
axs[1].set_ylabel("MAE")
axs[1].set_xlabel("Cells number")
axs[1].set_title("Distribution by bins")

fig.suptitle("Metric dependence of cells number")
plt.tight_layout()
plt.savefig("validation/MAE_distribution.png")
plt.show()


# IOU METRIC
values = []
labels = []
bins = np.arange(20, 50, 5)
for i in range(len(bins[:-1])):
    ts = df[(bins[i] <= df["cells_num"]) & (df["cells_num"] < bins[i + 1])][
        "IOU"
    ].values
    labels.append(f"{bins[i]}-{bins[i+1]}")
    values.append(ts)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].scatter(df["cells_num"], df["IOU"])
axs[0].set_ylabel("IOU")
axs[0].set_xlabel("Cells number")
axs[0].set_title("Each point is image")

axs[1].boxplot(values, tick_labels=labels)
axs[1].set_ylabel("IOU")
axs[1].set_xlabel("Cells number")
axs[1].set_title("Distribution by bins")

fig.suptitle("Metric dependence of cells number")
plt.tight_layout()
plt.savefig("validation/IoU_distribution.png")
plt.show()
