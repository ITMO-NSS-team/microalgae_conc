import os
import time

import pandas as pd

from CellCounter.ConcentrationCalculator import calculate_concentration
import cv2

from CellCounter.Segmentator import detect_cells, visualize_circles


def segment_sample(path_to_images: str,
                   dilution: float = 1,
                   image_volume: float = 0.0240,
                   dist: int = 10,
                   min_radius: int = 3,
                   max_radius: int = 20,
                   sensitivity: int = 20,
                   blur: int = 3):
    cells_df = pd.DataFrame()
    save_img_path = f'{path_to_images}/segmented'
    cells_num = []
    images_names = []
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    for file in os.listdir(path_to_images):
        print(f'Process {path_to_images}/{file}')
        if file not in ['segmented']:
            img = cv2.imread(f'{path_to_images}/{file}')
            cells = detect_cells(img,
                                 minDist=dist,
                                 minRadius=min_radius,
                                 maxRadius=max_radius,
                                 param2=sensitivity,
                                 blur_kernel=blur)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            visualize_circles(img, cells, f'{save_img_path}/seg_{file}')
            # visualize_circles(img, cells)
            cells_num.append(cells.shape[1])
            images_names.append(file.split('.')[0])
    concentrations = []
    for n in cells_num:
        cells_count = calculate_concentration(n, dilution, image_volume)
        concentrations.append(cells_count)

    cells_df['image'] = images_names
    cells_df['cells_num'] = cells_num
    cells_df['concentration'] = concentrations
    cells_df.to_csv(f'{path_to_images}/segmented/cells.csv', index=False)


samples = []
times = []

images_path = 'data'
conc_per_sample = []
for s in range(1, 16):
    start = time.time()
    folder = f'{images_path}/{s}'
    segment_sample(folder)
    end = time.time()
    times.append(end-start)
    samples.append(s)

times_df = pd.DataFrame()
times_df['Sample'] = samples
times_df['Time, s'] = times
times_df.to_csv('validation/auto_time.csv')

segment_sample('degradated_images', sensitivity=18, dist=20)

