from copy import deepcopy
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt


def shift_spectrum(image, channel_num: int):
    """
    Function for increasing contrast in one channel of an image using CLAHE
        :param image: Input BGR image as numpy array
        :param channel_num: Integer index of channel to enhance (0=Blue, 1=Green, 2=Red)
        :return: Image with enhanced channel as numpy array

    """

    chnls = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(chnls[channel_num])
    new_chs = []
    for i, c in enumerate(chnls):
        if i != channel_num:
            new_chs.append(c)
        else:
            new_chs.append(cl)
    img = cv2.merge(new_chs)
    return img


def visualize_circles(image, circles, save_path=None, annotate=False, ext_title=False):
    """
    Function to save plots with detected objects
        :param image: Input BGR image as numpy array
        :param circles: Numpy array of detected circles in format [[[x, y, radius], ...]]
        :param save_path: String with path where to save visualization (None to display)
        :param annotate: Boolean whether to number detected circles
        :param ext_title: Boolean whether to show extended title with area statistics
        :return: None (shows or saves plot)

    """

    if circles.shape[1] == 0:
        return None
    img_vis = deepcopy(image)
    plt.rcParams["figure.figsize"] = (8, 6)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img_vis, (i[0], i[1]), i[2], (255, 0, 0), 5)

    cells_area = np.round(np.sum(circles[0, :, 2] ** 2 * np.pi), 3)
    cells_mean_rad = int(np.round(np.mean(circles[0, :, 2])))
    if ext_title:
        plt.title(
            f"Full image cell number = {circles.shape[1]}\n"
            f"full cells area = {cells_area}\n"
            f"mean cell radius = {cells_mean_rad}"
        )
    else:
        plt.title(f"Full image cell number = {circles.shape[1]}")

    inds = np.arange(circles.shape[1])
    if annotate:
        for ind in inds:
            plt.annotate(
                ind, (circles[0, ind, 0], circles[0, ind, 1]), c="blue", fontsize=9
            )
    plt.imshow(img_vis)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100)
        plt.close()
    else:
        plt.show()


def return_mask(image, circles):
    """
    Function to create mask with black circles at detected locations
        :param image: Input BGR image as numpy array
        :param circles: Numpy array of detected circles in format [[[x, y, radius], ...]]
        :return: Mask image with black circles on white background as numpy array

    """

    img_vis = deepcopy(image)
    for i in circles[0, :]:
        cv2.circle(img_vis, (i[0], i[1]), i[2], (0, 0, 0), -1)
    return img_vis


def detect_cells(
    image,
    increase_channel=None,
    minDist=100,
    minRadius=15,
    maxRadius=100,
    param2=30,
    blur_kernel=25,
):
    """
    Function to detect circular cells using Hough Circle Transform
        :param image: Input BGR image as numpy array
        :param increase_channel: Integer index of channel to enhance (0=Blue, 1=Green, 2=Red)
        :param minDist: Minimum distance between detected circles in pixels
        :param minRadius: Minimum cell radius to detect in pixels
        :param maxRadius: Maximum cell radius to detect in pixels
        :param param2: Accumulator threshold (lower values detect more false circles)
        :param blur_kernel: Kernel size for median blur (must be odd integer)
        :return: Numpy array of detected circles in format [[[x, y, radius], ...]] or empty array

    """

    img = image
    if increase_channel is not None:
        img = shift_spectrum(img, increase_channel)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, blur_kernel)

    param1 = 30

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius,
    )
    if circles is None:
        print("Failed")
        return np.array([[]])

    circles = np.uint16(np.around(circles))
    return circles
