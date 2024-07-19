import numpy as np

from settings import Settings
from glob import glob


def get_bird_species():
    main_path = Settings.main_path

    if Settings.on_pycharm:
        get_species = lambda x: x.split('\\')[-1]
    else:
        get_species = lambda x: x.split('/')[-1]

    bird_species = [get_species(i) for i in glob(f'{main_path}/train_audio/*')]
    return bird_species

if len(get_bird_species())==0:
    print('Не тот компьютер')
    input('>>><<<')