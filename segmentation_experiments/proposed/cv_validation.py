import os
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cv_algorithms import detect_cell_cv2, detect_cv2_objs

images_path = "../data/raw"
masks_path = "../data/marked"


def iou(predicted, target):
    """
    Return intersection over union metric for two classes (classic)

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


def iou_one_class(predicted, target):
    """
    Return intersection over union metric for one class (only cells)

    """

    predicted = deepcopy(predicted)
    target = deepcopy(target)

    predicted[predicted != 0] = 1
    target[target != 0] = 1

    diff = predicted - target
    tp = np.sum(diff == 0)
    fp = np.sum(diff == 1)
    return np.round(tp / (tp + fp), 3)


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


def validate_cv_with_real(save_path):
    """
    No valid docstring found.
    """

    if not os.path.exists(save_path):
        os.makedirs(f"{save_path}/images")
    df = pd.DataFrame(
        columns=[
            "file_name",
            "cells_num",
            "MAE",
            "IOU",
            "IOU(cells)",
            "Cells area",
            "Cells area error %",
            "Cells area error pixels",
        ]
    )
    names = []
    cells_num = []
    maes = []
    ious = []
    ious_one = []
    areas = []
    areas_errors = []
    areas_errors_pixels = []

    for file in os.listdir(masks_path):
        mask = detect_cell_cv2(f"{images_path}/{file}", plot_visual=False)
        img = cv2.imread(f"{masks_path}/{file}")

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].imshow(mask, cmap="Grays")
        axs[0, 0].set_title("Detected")
        axs[0, 1].imshow(img, cmap="Grays")
        axs[0, 1].set_title("Real")

        mask_g = np.mean(mask, axis=2)
        mask_g[mask_g != 0] = 255
        img_g = np.mean(img, axis=2)
        img_g[img_g != 0] = 255

        mask_objs = detect_cv2_objs(mask_g)
        real_objs = detect_cv2_objs(img_g)

        mae_cells = abs(real_objs.shape[0] - mask_objs.shape[0])
        iou_val = iou(mask_g, img_g)
        iou_one_val = iou_one_class(mask_g, img_g)
        area_val = area_error(mask_g, img_g)
        area_pixel_val = area_error(mask_g, img_g, percent=False)

        axs[1, 0].imshow(mask, cmap="Grays")
        for p in mask_objs:
            axs[1, 0].scatter(p[0], p[1], c="r", s=5)
        axs[1, 0].set_title("Detected")
        axs[1, 1].imshow(img, cmap="Grays")
        for p in real_objs:
            axs[1, 1].scatter(p[0], p[1], c="r", s=5)
        axs[1, 1].set_title("Real")

        plt.suptitle(
            f"Cells MAE={mae_cells}, IOU={iou_val}, IOU(cells)={iou_one_val}, cells area error={area_val}"
        )

        plt.tight_layout()
        plt.savefig(f'{save_path}/images/{file.split(".")[0]}_val.png')
        plt.close()

        names.append(file.split(".")[0])
        cells_num.append(real_objs.shape[0])
        maes.append(mae_cells)
        ious.append(iou_val)
        ious_one.append(iou_one_val)

        areas.append(np.sum(img_g == 0))
        areas_errors.append(area_val)
        areas_errors_pixels.append(area_pixel_val)

    df["file_name"] = names
    df["cells_num"] = cells_num
    df["MAE"] = maes
    df["IOU"] = ious
    df["IOU(cells)"] = ious_one
    df["Cells area"] = areas
    df["Cells area error %"] = areas_errors
    df["Cells area error pixels"] = areas_errors_pixels

    df.to_csv(f"{save_path}/metrics.csv", index=False)


# out_folder = f'validation'
# validate_cv_with_real(out_folder)
