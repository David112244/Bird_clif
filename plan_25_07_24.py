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

from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping

import models
import small_functions as sf
import medium_functions as mf
import settings

bird_species = settings.get_bird_species()
main_path = settings.Settings.main_path


def marking(bird_id):
    # каждый вид в отдельную папку
    # каждую запись в отдельную папку
    # в начале и конце каждой папки записи создавать по три пустые записи
    paths = sf.get_bird_paths(bird_id, 'train')
    species = bird_species[bird_id]
    present_files = glob(f'{main_path}/marking_spectrogram/{species}/*')
    while 1:
        path = paths[np.random.randint(0, len(paths))]
        if path in present_files:
            continue
        audio, sr = lb.load(path)
        length = len(audio) / sr
        if length < 2.5:
            continue
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

        audio_folder_name = sf.get_file_name(path)
        os.makedirs(f'{main_path}/marking_spectrogram/{species}/{audio_folder_name}', exist_ok=True)
        count_files_in_folder = len(glob(f'{main_path}/marking_spectrogram/{species}/{audio_folder_name}/*'))
        if count_files_in_folder == 0:
            for i in range(3):
                sf.add_empty_spectrogram(
                    f'{main_path}/marking_spectrogram/{species}/{audio_folder_name}/{i}_segment_0_auto.png')
                count_files_in_folder += 1
        segments = sf.split_spectrogram(settings.Settings.count_slices_in_step, main_spec)
        portions = sf.return_segments_for_plots(segments)
        for p in portions:
            sf.create_plots(main_spec, path, p, [i for i in range(16)])
            answers = sf.input_marking_answers(len(p))

            if np.array_equal(answers, np.array(['s'])):
                for _ in range(3):
                    sf.add_empty_spectrogram(
                        f'{main_path}/marking_spectrogram/{species}/{audio_folder_name}/'
                        f'{count_files_in_folder}_segment_0_auto.png')
                    count_files_in_folder += 1
                return

            for seg, ans in zip(p, answers):
                cv2.imwrite(
                    f'{main_path}/marking_spectrogram/{species}/{audio_folder_name}/'
                    f'{count_files_in_folder}_segment_{ans}_manual.png', seg * 255)
                count_files_in_folder += 1


def load_data(bird_id):
    species = bird_species[bird_id]
    all_paths = []
    for folder in glob(f'{main_path}/marking_spectrogram/{species}/*'):
        for path in glob(f'{folder}/*'):
            all_paths.append(path)
    features = []
    targets = []
    for p in all_paths:
        targets.append(int(p.split('_')[-2]))
        features.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE) / 255)
    return sf.create_batch(features, targets, 3)


def check_accuracy_load_data(f, t):
    for f_, t_ in zip(f, t):
        print(t_)
        lb.display.specshow(f_[1])
        plt.show()


def learn_model(bird_id, features, targets):
    species = bird_species[bird_id]
    model = models.model_7(2, 3)
    tar = to_categorical(targets)
    early_stopping = EarlyStopping(
        min_delta=0.01,
        verbose=1,
        patience=10,
        start_from_epoch=5,
        restore_best_weights=True
    )

    model.fit(
        features,
        tar,
        validation_split=0.1,
        epochs=500,
        batch_size=8,
        steps_per_epoch=features.shape[0] // 8,
        callbacks=[early_stopping]
    )
    model.save(f'{main_path}/models/model_7_marking_{species}.keras')


def check_accuracy_first_learn_model(bird_id):
    species = bird_species[bird_id]
    check_paths = glob(f'{main_path}/marking_spectrogram/{species}/*')
    folder_names = np.array([check_path.split(f'\\')[-1] for check_path in check_paths])
    np.random.shuffle(folder_names)
    features = []
    for name in folder_names:
        paths_to_segments = glob(f'{main_path}/marking_spectrogram/{species}/{name}/*')
        for path in paths_to_segments:
            features.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    features, _ = sf.create_batch(features, [1 for i in range(len(features))])
    features = features.reshape(features.shape[0], 3, 256, 256, 1)

    model = load_model(f'{main_path}/models/model_9_marking_{species}.keras')
    prediction = np.round(model.predict(features), 3)
    features_to_show = np.squeeze([i[1] for i in features])
    features_to_show = sf.return_segments_for_plots(features_to_show)
    prediction_to_show = sf.return_segments_for_plots(prediction)
    for f, p in zip(features_to_show, prediction_to_show):
        sf.create_plots(f[0], species, f, p)
        sleep(3)
        input('Go?')


# f, t = load_data(0)
# print(f.shape, t.shape)
# check_accuracy_load_data(f, t)
