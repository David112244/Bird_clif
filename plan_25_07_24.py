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

# from keras.utils import to_categorical
# from keras.models import load_model
# from keras.callbacks import EarlyStopping
#
# import models
import small_functions as sf
import medium_functions as mf
import settings

bird_species = sf.get_bird_species()
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
    for name in folder_names:
        path = f'{main_path}/train_audio/{species}/{name}.ogg'
        audio, sr = lb.load(path)
        length = len(audio) / sr
        if length < 2.5:
            continue
        count_slices = int(length * settings.Settings.count_slices_in_sec)
        main_spec = sf.spec_from_audio(audio, count_slices, 256)

        segments = sf.split_spectrogram(256, main_spec)
        # sfp = sf.return_segments_for_plots(segments)

        features, _ = sf.create_batch(segments, [1 for i in range(len(segments))])
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


def check_marking_data():
    folder_names = glob(f'{main_path}/output_data/*')
    np.random.shuffle(folder_names)
    for folder in folder_names:
        paths = glob(f'{folder}/*')
        specs = []
        labels = []
        for path in paths:
            file_name = path.split('\\')[-1]
            labels.append(file_name.split('_')[2])
            specs.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        specs_batch = sf.return_segments_for_plots(specs)
        labels_batch = sf.return_segments_for_plots(labels)
        paths_batch = sf.return_segments_for_plots(paths)
        for s, l, p, i in zip(specs_batch, labels_batch, paths_batch, range(len(specs_batch))):
            if i >= 3: break
            i += 1
            sf.create_plots(s[0], folder, s, l)
            inp = input('Norm?')
            nums = [p_.split('\\')[-1].split('_')[0] for p_ in p]
            if inp == 'n':
                answers = sf.input_marking_answers(len(s))
                [os.remove(pt) for pt in p]
                for new_l, seg, num in zip(answers, s, nums):
                    cv2.imwrite(f'{folder}/{num}_segment_{new_l}_remarkable.png', seg)
            elif inp == 's':
                break
            else:
                [os.remove(pt) for pt in p]
                for seg, leb, num in zip(s, l, nums):
                    cv2.imwrite(f'{folder}/{num}_segment_{leb}_modelCheck.png', seg)


def add_noise(spec, intensity=10000):
    s = spec.copy()
    row = np.random.randint(0, 256, intensity)
    col = np.random.randint(0, 256, intensity)
    for r, c in zip(row, col):
        s[r][c] = np.random.randint(0, 255)
    return s


def brightness_change(spec, intensity):
    return spec - intensity


def contrast(spec):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(spec, -1, kernel=kernel)


def create_learn_data(bird_id):
    # path = r'E:\datas\birdclif\marking_spectrogram\asbfly\XC305518/16_segment_1_manual.png'
    # spec = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # fig, axes = plt.subplots(1, 5)
    # lb.display.specshow(spec, ax=axes[0])
    # lb.display.specshow(add_noise(spec, 30000), ax=axes[1])
    # lb.display.specshow(brightness_change(spec, 200), ax=axes[2])
    # lb.display.specshow(brightness_change(spec, 50), ax=axes[3])
    # lb.display.specshow(contrast(spec), ax=axes[4])
    # plt.show()
    species = bird_species[bird_id]
    folder_names = glob(f'{main_path}/marking_spectrogram/{species}/*')
    for folder_path in folder_names:
        paths = glob(f'{folder_path}/*')
        folder_name = folder_path.split('\\')[-1]
        sort_paths = sf.sort_segments(paths, delimiter='\\')
        graduation = ['_unchanged', '_plus', '_minus', '_contrast', '_noise']
        for g in graduation:
            os.makedirs(f'{main_path}/to_input_data/{species}/{folder_name}{g}', exist_ok=True)
        # for g in graduation:
        #     for i in range(3):
        #         sf.add_empty_spectrogram(f'{main_path}/to_input_data/{species}/{folder_name}{g}/'
        #                                  f'{i}_segment_0_auto.png')
        count_file_in_folder = 0
        for path in sort_paths:
            label = path.split('_')[-2]
            unchanged = cv2.imread(path)
            plus = brightness_change(unchanged, 200)
            minus = brightness_change(unchanged, 50)
            cont = contrast(unchanged)
            noise = add_noise(unchanged, 30000)

            variants = [unchanged, plus, minus, cont, noise]
            for g, v in zip(graduation, variants):
                cv2.imwrite(f'{main_path}/to_input_data/{species}/{folder_name}{g}/'
                            f'{count_file_in_folder}_segment_{label}_manualvar.png', v)
            count_file_in_folder += 1


# create_learn_data(0)
