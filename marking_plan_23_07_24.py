import os
import pandas as pd
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt

from glob import glob
import cv2
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping

import models
import small_functions as sf
import settings

bird_species = settings.get_bird_species()[:]
main_path = settings.Settings.main_path
print(len(bird_species))


def load_data():
    features = []
    targets = []
    bs = bird_species
    np.random.shuffle(bs)
    for species in bs[:5]:
        i = sf.get_index(species)
        print(species)
        folder_path = f'{main_path}/spectrogram/{species}'
        os.makedirs(folder_path, exist_ok=True)
        species_paths = sf.get_bird_paths(i, 'train', test_size=0.5)
        np.random.shuffle(species_paths)
        print(len(species_paths))
        for i, path in enumerate(species_paths[:10]):
            print(i, end=' ')
            if settings.Settings.on_pycharm:
                file_name = path.split('\\')[-1].split('.')[0]
            else:
                file_name = path.split('/')[-1].split('.')[0]
            os.makedirs(f'{folder_path}/{file_name}', exist_ok=True)

            audio, sr = lb.load(path)
            length = len(audio) / sr
            if length < 2.5:
                continue
            count_slices = int(length * settings.Settings.count_slices_in_sec)
            main_spec = sf.spec_from_audio(audio, count_slices, 256)
            segments = sf.split_spectrogram(settings.Settings.count_slices_in_step, main_spec)
            np.random.shuffle(segments)
            for seg in segments[:10]:
                features.append(seg)
                targets.append(i)
    return [np.array(features), np.array(targets)]


def load_data_from_drive_to_colab(count_in_species):
    data = []
    targets = []
    index = 0
    for i, species in enumerate(bird_species):
        species_paths = glob(f'{main_path}/spectrogram/{species}/*')
        all_paths = species_paths
        # for audio_folder in species_paths:
        #     segments = glob(f'{audio_folder}/*')
        #     for seg in segments:
        #         all_paths.append(seg)

        np.random.shuffle(all_paths)
        for path in all_paths[:count_in_species]:
            print(index)
            index += 1
            s = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if np.array(s).shape==(256,256):
                data.append(s)
                targets.append(i)
    return [np.array(data), np.array(targets)]


def learn_model():
    model = models.model_8(out=182)
    for _ in range(100):
        features, targets = load_data()
        features = features.reshape(features.shape[0], 256, 256, 1)
        targets = to_categorical(targets, num_classes=182)

        early_stopping = EarlyStopping(
            min_delta=0.01,
            verbose=1,
            patience=5,
            # start_from_epoch=25,
            restore_best_weights=True
        )
        model.fit(
            features, targets,
            validation_split=0.1,
            steps_per_epoch=features.shape[0] // 64,
            batch_size=64,
            epochs=1000,
            callbacks=[early_stopping]
        )
        model.save(f'{main_path}/models/model_8_raw_marking.keras')


def check_accuracy_model():
    model = load_model(f'{main_path}/models/model_8_raw_marking.keras')
    for _ in range(10):
        species = bird_species[np.random.randint(0, 10)]
        species_index = sf.get_index(species)
        species_paths = sf.get_bird_paths(species_index, 'test', test_size=0.5)
        batch_species_paths = species_paths[np.random.randint(0, len(species_paths), 16)]
        batch = []
        for path in batch_species_paths:
            audio, sr = lb.load(path)
            length = len(audio) / sr
            count_slices = int(length * settings.Settings.count_slices_in_sec)
            main_spec = sf.spec_from_audio(audio, count_slices, 256)
            segments = sf.split_spectrogram(settings.Settings.count_slices_in_step, main_spec)
            seg = segments[np.random.randint(0, len(segments))]
            batch.append(seg)

        batch = np.array(batch)
        features = batch.reshape(16, 256, 256, 1)
        prediction = model.predict(features)
        # str_predictions = [','.join([str(np.round(p, 2)) for p in pr]) for pr in prediction]
        difference_prediction = [str(1 - p[species_index]) for p in prediction]

        sf.create_plots(batch[0], f'Must be {species_index}', batch, difference_prediction)


def check_accuracy_model_using_load_data():
    model = load_model(f'{main_path}/models/model_8_raw_marking.keras')
    for _ in range(10):
        features, targets = load_data_from_drive_to_colab(16)
        p_features = sf.return_segments_for_plots(features)
        p_targets = sf.return_segments_for_plots(targets)
        for f, t in zip(p_features, p_targets):
            features = f.reshape(16, 256, 256, 1)
            prediction = model.predict(features)
            # str_predictions = [','.join([str(np.round(p, 2)) for p in pr]) for pr in prediction]
            difference_prediction = [str(1 - p[tt]) for p,tt in zip(prediction,t)]

            sf.create_plots(f[0], f'Must be', f, difference_prediction)


def create_spec_with_drive_to_drive():
    for species in bird_species[:10]:
        print(species)
        bird_index = sf.get_index(species)
        paths = sf.get_bird_paths(bird_index, 'train', 0.5)[:50]
        index = 0
        os.makedirs(f'{main_path}/spectrogram/{species}')
        for i, path in enumerate(paths):
            print(i)
            audio, sr = lb.load(path)
            length = len(audio) / sr
            if length < 2.5:
                continue
            count_slices = int(length * settings.Settings.count_slices_in_sec)
            main_spec = sf.spec_from_audio(audio, count_slices, 256)
            segments = sf.split_spectrogram(256, main_spec)
            for seg in segments:
                cv2.imwrite(f'{main_path}/spectrogram/{species}/segment_{index}.png', seg * 255)
                index += 1
# check_accuracy_model()
