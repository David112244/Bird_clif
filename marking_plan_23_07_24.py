import os
import pandas as pd
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt

from glob import glob
import cv2

import small_functions as sf
import medium_functions as mf
import settings

output = 5
bird_species = settings.get_bird_species()[1:output]
main_path = settings.Settings.main_path


def create_spectrogram_each_spectrogram():
    for i, species in enumerate(bird_species):
        i+=1
        print(i)
        folder_path = f'{main_path}/spectrogram/{species}'
        os.makedirs(folder_path, exist_ok=True)
        for path in sf.get_bird_paths(i, 'train'):
            print(path)
            file_name = path.split('\\')[-1].split('.')[0]
            os.makedirs(f'{folder_path}/{file_name}', exist_ok=True)

            audio, sr = lb.load(path)
            length = len(audio) / sr
            if length<2.5:
                continue
            count_slices = int(length * settings.Settings.count_slices_in_sec)
            main_spec = sf.spec_from_audio(audio, count_slices, 256)
            segments = sf.split_spectrogram(settings.Settings.count_slices_in_step, main_spec)
            file_path = f'{folder_path}/{file_name}'
            index = 0
            # for _ in range(3):
            #     sf.add_empty_spectrogram(file_path, f'segment_{index}.jpg')
            #     index += 1
            for seg in segments:
                cv2.imwrite(f'{file_path}/segment_{index}.jpg', seg)
                index += 1
            # for _ in range(3):
            #     sf.add_empty_spectrogram(file_path, f'segment_{index}.jpg')
            #     index += 1


create_spectrogram_each_spectrogram()
