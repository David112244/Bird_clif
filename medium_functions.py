import os

import numpy as np
import pandas as pd

import librosa as lb
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import seaborn as sns
from collections import Counter

# from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical
# from keras.models import load_model

from glob import glob
from time import time

import small_functions as sf
from settings import Settings

# import models

main_path = 'E:/datas/birdclif'
get_species = lambda x: x.split('\\')[-1]
bird_species = np.array([get_species(i) for i in glob(f'{main_path}/train_audio/*')])


# создаёт тренировочный набор данных из спектрограм
def create_train_frame(train_paths):
    print('\nCreate_train_frame\n')
    train_array = []
    for_marking = []
    for ind, species in enumerate(bird_species):
        print(f'{ind + 1} of {len(bird_species)}')
        paths_species_file = []
        for path in train_paths:
            if path.split('\\')[0].split('/')[-1] == species:
                paths_species_file.append(path)
        used_index = []
        used_index.append(np.random.randint(0, len(paths_species_file)))
        current_file = paths_species_file[used_index[-1]]
        segments = sf.split_audio(current_file)
        current_index = 0
        for i in range(Settings.count_each_species):
            try:
                seg = segments[i - current_index]
                spec = sf.spec_from_audio(seg, 256, 256)
                lb.display.specshow(spec)
                plt.show()
                train_array.append(spec)
                for_marking.append(species)
            except IndexError:
                for _ in range(10):
                    index = np.random.randint(0, len(paths_species_file))
                    if index in used_index:
                        continue
                    else:
                        used_index.append(index)
                        current_file = paths_species_file[used_index[-1]]
                        segments = sf.split_audio(current_file)
                        current_index = i

                        seg = segments[i - current_index]
                        spec = sf.spec_from_audio(seg, 256, 256)
                        train_array.append(spec)
                        for_marking.append(species)
                        break
    train_array = np.array(train_array)
    return [train_array, for_marking]


def learn_model(train_array, test_array, is_first=True):
    features = train_array.reshape(train_array.shape[0], 256, 256, 1)
    # lb.display.specshow(features[3].reshape(128, 256))
    # plt.show()
    targets = to_categorical(test_array)
    x_train, x_val, y_train, y_val = train_test_split(features, targets, test_size=0.2)

    if is_first:
        model = models.model3(len(targets[0]))
    else:
        model = load_model(f'{main_path}/best_models/model_3')

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=x_train.shape[0] // 50,
        steps_per_epoch=50,
        epochs=20
    )
    model.save(f'{main_path}/best_models/model_3')


def go(train, is_first):
    train_array, for_marking = create_train_frame(train)
    test_array = sf.marking(for_marking)
    learn_model(train_array, test_array, is_first)
    is_first = False


def train_model(train, epochs, is_first=True):
    for i in range(epochs):
        print(f'\nEpoch №{i + 1}\n')
        if is_first:
            go(train, is_first)
            is_first = False
        else:
            go(train, is_first)


# функция проверяющиая точность предсказания
def check_accuracy(test_paths):
    model = load_model(f'{main_path}/best_models/model_3')
    accuracy_score = []
    for j, path in enumerate(test_paths):
        print(f'Path {j} in {len(test_paths)}')
        label = sf.get_index(path.split('\\')[0].split('/')[-1])
        segments = sf.split_audio(path)
        test_array = np.array([sf.spec_from_audio(seg, 256, 256) for seg in segments])
        test_array = test_array.reshape(test_array.shape[0], 256, 256, 1)
        answers = model.predict(test_array, verbose=0)
        answer = np.argmax(np.sum(answers, axis=0))

        if answer == label:
            accuracy_score.append(1)
            print('True')
        else:
            accuracy_score.append(0)
            print('False')
    return np.mean(accuracy_score)


def create_spectrogram_each_species():
    for species in bird_species:
        print(species)
        paths = glob(f'{main_path}/train_audio/{species}/*')
        path = paths[np.random.randint(0, len(paths))]
        segments = sf.split_audio(path)
        seg = segments[np.random.randint(0, len(segments))]
        spec = sf.spec_from_audio(seg, 256, 256)
        pd.DataFrame(spec).to_csv(f'{main_path}/each_species_spectrogram/{species}_2.csv')



# create_spectrogram_each_species()
# train_data, test_data = sf.train_test_rate()
# train_model(train_data, 200)
# print(check_accuracy(test_data))

#