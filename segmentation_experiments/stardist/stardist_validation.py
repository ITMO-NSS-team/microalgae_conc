import os
from copy import deepcopy

import pandas as pd
from matplotlib import pyplot as plt
from stardist.models import StarDist2D
import numpy as np
from matplotlib.image import imread
from skimage.transform import resize


"""
Pre-trained models on stardist algorithm used from repo
https://github.com/stardist/stardist/tree/main
"""


def iou(predicted, target):
    """
    No valid docstring found.
    """

    predicted = deepcopy(predicted)
    target = deepcopy(target)

    predicted[predicted != 0] = 1
    target[target != 0] = 1
    diff = predicted - target
    tp = np.sum(diff == 0)
    fp = np.sum(diff == 1)
    fn = np.sum(diff == -1)
    return np.round(tp / (tp + fp + fn), 3)


def area_error(predicted, target, percent=True):
    """
    No valid docstring found.
    """

    predicted = deepcopy(predicted)
    target = deepcopy(target)

    predicted[predicted != 0] = 1
    target[target != 0] = 1
    if percent:
        return np.round(abs(np.sum(predicted) - np.sum(target)) / np.sum(target), 3)
    else:
        return abs(np.sum(predicted) - np.sum(target))


inner_size = (3024, 4032)
grayscale_models = ["2D_versatile_fluo", "2D_paper_dsb2018"]

# model_name = '2D_versatile_he'
model_name = "2D_paper_dsb2018"

files_folder = "../data/raw"
target_folder = "../data/marked"

names = []
cells_nums = []
iuo_errors = []
mae_errors = []
areas_errors = []

for file in os.listdir(files_folder):
    save_folder = f"stardist_validation/{model_name}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    img = imread(f"{files_folder}/{file}")
    data = resize(img, (512, 512, 3))
    if model_name in grayscale_models:
        # data = np.mean(data, axis=2)
        data = data[:, :, 0] * 0.299 + data[:, :, 1] * 0.587 + data[:, :, 2] * 0.114

    model = StarDist2D.from_pretrained(model_name)
    labels, details = model.predict_instances(data, prob_thresh=0.4)
    coord, points, prob = details["coord"], details["points"], details["prob"]

    labels[labels != 0] = 1
    labels = resize(labels, inner_size)
    labels[labels != 0] = 1

    """plt.imshow(img)
    #_draw_polygons(coord, points, prob, show_dist=False)
    plt.imshow(labels, cmap='Grays', alpha=0.5)
    #plt.title('p=0.4')
    plt.axis('off')
    plt.tight_layout()
    plt.show()"""

    target_cells = pd.read_csv("../validation/metrics.csv")
    target_num = target_cells[target_cells["file_name"] == file.split(".")[0]][
        "cells_num"
    ].item()

    marked_img = imread(f"{target_folder}/{file}")[:, :, 0]
    marked_img[marked_img != 0] = 1
    marked_img = 1 - marked_img

    iou_error = iou(labels, marked_img)
    mae_error = abs(points.shape[0] - target_num)
    area_val = area_error(labels, marked_img)

    names.append(file.split(".")[0])
    iuo_errors.append(iou_error)
    mae_errors.append(mae_error)
    cells_nums.append(target_num)
    areas_errors.append(area_val)

    labels[labels != 1] = None

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].imshow(img, cmap="Grays")
    axs[1].imshow(img)
    axs[1].imshow(labels, cmap="Greys_r")
    axs[0].axis("off")
    axs[1].axis("off")
    axs[0].set_title("Real image")
    axs[1].set_title("Cells mask")
    plt.suptitle(
        f"StarDist({model_name}), \ntarget cells number = {target_num}, IoU={iou_error}, MAE={mae_error}"
    )
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{file}")
    plt.close()

    df = pd.DataFrame()
    df["file_name"] = names
    df["cells_num"] = cells_nums
    df["MAE"] = mae_errors
    df["IOU"] = iuo_errors
    df["Cells area error %"] = areas_errors
    df.to_csv(f"stardist_validation/{model_name}.csv", index=False)
