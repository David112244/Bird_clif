import numpy as np


class Settings:
    seed = None
    debug = False
    count_slices_in_sec = 128
    count_slices_in_step = 256
    full = False
    on_pycharm = True

    if on_pycharm:
        main_path = 'E:/datas/birdclif'
    else:
        main_path = '/content/drive/MyDrive/bird_clif'


def get_bird_species():
    main_path = Settings.main_path

    if Settings.on_pycharm:
        get_species = lambda x: x.split('\\')[-1]
    else:
        get_species = lambda x: x.split('/')[-1]

    bird_species = [get_species(i) for i in glob(f'{main_path}/train_audio/*')]
    return bird_species


print(Settings.main_path)
