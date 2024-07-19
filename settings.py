import numpy as np


class Settings:
    seed = None
    debug = False
    count_slices_in_sec = 128
    count_slices_in_step = 256
    full = False
    on_pycharm = False

    if on_pycharm:
        main_path = 'E:/datas/birdclif'
    else:
        main_path = '/content/drive/MyDrive/bird_clif'


print(Settings.main_path)
