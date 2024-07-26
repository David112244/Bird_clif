# какие мне нужны функции:
# 1. разметки
# 2. проверки проверка точности недоразмеченных данных
import os
from typing import List

import pandas as pd
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt

import sys
from glob import glob
from time import sleep
from collections import Counter
import cv2

# import models
import small_functions as sf
import medium_functions as mf
import settings

output = 183
bird_species = settings.get_bird_species()
main_path = settings.Settings.main_path


# проверить как лучше сохранять спектрограммы в картинки или в .csv
def marking(bird_id):
    # каждый вид в отдельную папку
    # между записями создавать три пустые спектрограмы (с нулями)
    paths = sf.get_bird_paths(bird_id, 'train')
    species = bird_species[bird_id]
    # if len(paths_in_species) == 0:
    #     for i in range(3):
    #         sf.add_empty_spectrogram(species)
    # elif len(paths_in_species) > 3 and np.mean(cv2.imread(paths_in_species[-1])) == 0:
    #     for p in paths_in_species[-3:]:
    #         os.remove(p)
    while 1:
        path = paths[np.random.randint(0, len(paths))]
        if settings.Settings.on_pycharm:
            folder_name = path.split('\\')[-1].split('.')[0]
        else:
            folder_name = path.split('/')[-1].split('.')[0]
        # folder_name = 'XC175797'
        folder_path = f'{main_path}/marking_spectrogram/{species}/{folder_name}'

        audio, sr = lb.load(path)
        length = len(audio) / sr
        count_slices = int(length * settings.Settings.count_slices_in_sec)
        main_spec = sf.spec_from_audio(audio, count_slices, 256)

        lb.display.specshow(main_spec)
        plt.show()
        sleep(1)
        go = input('Marking? >>>')
        if go != 'g':
            print('No')
            continue
        print(path)
        os.makedirs(folder_path, exist_ok=True)

        paths_in_folder = glob(f'{folder_path}/*')
        if len(glob(f'{folder_path}/*')) == 0:
            for _ in range(3):
                sf.add_empty_spectrogram(species, folder_name)
        elif len(paths_in_folder) > 3 and np.mean(cv2.imread(paths_in_folder[-1])) == 0:
            for p in paths_in_folder[-3:]:
                os.remove(p)

        segments = sf.split_spectrogram(settings.Settings.count_slices_in_step, main_spec)
        portions = sf.return_segments_for_plots(segments)
        for p in portions:
            sf.create_plots(main_spec, path, p, [i for i in range(16)])
            answers = sf.input_marking_answers(len(p))
            if np.array_equal(answers, np.array(['s'])):
                for _ in range(3):
                    sf.add_empty_spectrogram(species, f'{main_path}/marking_spectrogram/{species}/{folder_name}/')
                return
            count_files = len(glob(f'{folder_path}/*'))
            for pic, an, i in zip(p, answers, [j for j in range(count_files, count_files + 8)]):
                file_name = f'{i}_batch_{an}_manually.png'
                cv2.imwrite(f'{folder_path}/{file_name}', pic)


marking(1)
