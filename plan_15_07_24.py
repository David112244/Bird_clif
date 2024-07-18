"""
1. Цикл предобработки данных
    - беру случайную запись каждого из видов
    - беру случайный отрезок из записи и режу на спектрограммы(есть функция)
    - убираю первую и последнюю спектрограммы
    - делаю прогноз с помощью леса
    - закидываю данные спектрограммы в модель
"""
import pandas as pd
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt

import sys
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical

import models
import small_functions as sf
import medium_functions as mf
from settings import Settings
from marking import create_plots

from joblib import dump, load

output = 182
bird_species = mf.bird_species[:output]
main_path = mf.main_path


def preparation_loop():
    train_spec_arr = []
    train_digi_arr = []
    train_targets = []
    for i in range(len(bird_species)):
        print(f'Bird index: {i}')
        paths = sf.get_bird_paths(i, 'train')
        for _ in range(2):
            path = paths[np.random.randint(0, len(paths))]

            length = 10
            audio, sr = lb.load(path)
            audio = sf.chop_audio(audio, sr, length)
            count_slices = length * Settings.count_slices_in_sec
            spectrogram = sf.spec_from_audio(audio, count_slices, 256)
            segments = sf.split_spectrogram(256, spectrogram)[1:-1]
            digital_data = np.array([sf.get_digital_data_from_spec(spec) for spec in segments])

            for s, d in zip(segments, digital_data):
                train_spec_arr.append(s)
                train_digi_arr.append(d)
                train_targets.append(i)
    return [np.array(train_spec_arr), np.array(train_digi_arr), np.array(train_targets)]


def learn_model():
    t_specs, t_digi, t_targest = preparation_loop()

    # forest = load(f'{main_path}/models/marking_forest')
    # add_features = np.array(forest.predict(t_digi))
    features = t_specs.reshape(t_specs.shape[0], 256, 256, 1)
    targets = to_categorical(t_targest, num_classes=len(bird_species))
    del t_specs, t_digi, t_targest

    x_train, x_val, y_train, y_val = train_test_split(features, targets, test_size=0.1)
    print('\nlearn model\n')
    model = models.model_5(out=output)
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=64,
        steps_per_epoch=x_train.shape[0] // 128+1,
        epochs=50
    )
    model.save(f'{main_path}/models/model_5')


learn_model()
input('><')