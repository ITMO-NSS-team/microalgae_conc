import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def calc_volume(
    imgs_path: str, l: float, h: float, P_h: int, P_w: int, plot_stats_path=None
):
    """
    Calculates the volume of a region within a microscope image, enabling accurate cell concentration measurements. It determines this volume based on chamber dimensions, object size in pixels, and calibration images of a grid pattern.

        :param imgs_path: Path to directory containing chamber grid calibration images.
               Images should show at least 5 squares in different chamber regions.
        :type imgs_path: str

        :param l: Known physical length of chamber square edge (mm).
               Serves as etalon for pixel-to-mm conversion (Step 1).
        :type l: float

        :param h: Chamber depth (mm). Critical for Z-dimension in volume calculation (Step 2).
        :type h: float

        :param P_h: Object height in pixels (from microscope image).
               Used in volume calculation V_img = (s·P_H)·(s·P_W)·h (Eq.2).
        :type P_h: int

        :param P_w: Object width in pixels (from microscope image).
               Second dimension for area calculation in volume formula (Eq.2).
        :type P_w: int

        :param plot_stats_path: If str, displays measurement statistics visualization:
               - Data points (red) showing all measured pixel values
               - Mean line (green) with conversion factor s = l/p_mean
               - Percentage differences from mean (gray dotted lines)
        :type plot_stats_path: str
        :default plot_stats_path: None

        :return: Image volume V_img in mL (Eq.2). Used to convert cell counts to concentration
                 via N = (a_img * 10³)/V_img (Eq.3).
        :rtype: float


        Note: Requires 5+ calibration images of chamber grid squares at different positions
        to establish reliable statistics. Volume calculation is specific to:
        - Chamber type
        - Microscope magnification
        - Camera resolution

    """

    p_list = []
    for file in os.listdir(imgs_path):
        img = cv2.imread(f"{imgs_path}/{file}")
        p_list.append(img.shape[0])
        p_list.append(img.shape[1])

    p_list = sorted(p_list)

    print(f"Standard deviation: {int(np.std(p_list))} px")

    p = round(np.mean(p_list))
    s = l / p
    v_img = (s * P_h) * (s * P_w) * h

    if plot_stats_path is not None:
        plt.rcParams["figure.figsize"] = (9, 5)  # Slightly larger for annotation

        # Main plot
        plt.scatter(
            np.arange(len(p_list)), p_list, c="r", label="Measured pixel values"
        )
        plt.axhline(
            p, c="green", linestyle="--", label=f"Mean = {p:.0f} px (s = {s:.1e} mm/px)"
        )

        # Difference lines
        plt.plot([], [], "gray", linestyle=":", alpha=0.5, label="Δ from mean (%)")
        for i, p_i in enumerate(p_list):
            diff = (p_i - p) / p * 100
            mid_y = (p + p_i) / 2
            plt.vlines(x=i, ymin=p, ymax=p_i, colors="gray", linestyles=":", alpha=0.5)
            plt.annotate(
                f"{diff:+.1f}%",
                xy=(i, mid_y),
                ha="right",
                va="center",
                rotation=90,
                fontsize=9,
                c="black",
            )

        # INFO BOX - Bottom right
        info_text = (
            f"Input Parameters:\n"
            f"- Chamber edge (l) = {l} mm\n"
            f"- Chamber depth (h) = {h} mm\n"
            f"- Object dims = {P_h}×{P_w} px\n\n"
            f"Conversion:\n"
            f"s = l/p_mean = {l}/{p} = {s:.1e} mm/px\n\n"
            f"Volume Calculation:\n"
            f"V_img = (s·P_H)·(s·P_W)·h\n"
            f"      = ({s:.1e}·{P_h})·({s:.1e}·{P_w})·{h}\n"
            f"      = {v_img:.4f} mL"
        )

        plt.gcf().text(
            0.97,
            0.10,
            info_text,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
            fontfamily="monospace",
        )

        plt.title("Microscope Image Volume Calibration")
        plt.ylabel("Pixel measurements")
        plt.legend(loc="upper left")

        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig(f"{plot_stats_path}")
        plt.close()

    return v_img
