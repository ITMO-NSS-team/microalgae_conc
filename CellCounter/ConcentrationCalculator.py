
def calculate_concentration(N: int, D: float, v_img: float):
    """
    Function for calculation cells concentration for standard chamber

    :param N: number of cells on photo
    :param D: dilution factor
    :param v_img: volume of image (mm^3)
    """
    cells_count = (N*D)/v_img  # mm^3
    cells_count = cells_count * 1000  # ml
    return round(cells_count)
