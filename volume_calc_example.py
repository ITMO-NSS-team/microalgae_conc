from CellCounter.VolumeCalculator import calc_volume

volume2 = calc_volume(
    "squares_statistic/eq2", 0.2, 0.1, 2592, 1944, "validation/squares_stat_report.png"
)

volume1 = calc_volume(
    "squares_statistic/eq1",
    0.05,
    0.1,
    4032,
    3024,
    "validation/squares_stat_report1.png",
)
