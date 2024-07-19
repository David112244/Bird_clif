import pandas as pd
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt

import sys
from glob import glob
from time import sleep
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping

import main
import models
import small_functions as sf
import medium_functions as mf
from settings import Settings

from joblib import dump, load

output = 182
bird_species = main.get_bird_species()
main_path = Settings.main_path

target_paths = ['train_audio/asbfly/XC856776.ogg',
                'train_audio/asbfly/XC600171.ogg']


# проверить как лучше сохранять спектрограммы в картинки или в .csv
def marking(bird_id):
    paths = sf.get_bird_paths(bird_id, 'train')
    while 1:
        path = paths[np.random.randint(0, len(paths))]
        audio, sr = lb.load(path)
        length = len(audio) / sr
        count_slices = int(length * Settings.count_slices_in_sec)
        main_spec = sf.spec_from_audio(audio, count_slices, 256)

        lb.display.specshow(main_spec)
        plt.show()
        sleep(1)
        go = input('Marking? >>>')
        if go != 'g':
            print('No')
            continue
        print(path)

        segments = sf.split_spectrogram(Settings.count_slices_in_step, main_spec)
        portions = sf.return_segments_for_plots(segments)
        for p in portions:
            sf.create_plots(main_spec, path, p, [i for i in range(16)])
            answers = sf.input_marking_answers(len(p))
            p = p.reshape(16, -1)
            df = pd.DataFrame(p)
            df['label'] = answers
            count = len(glob(f'{main_path}/marking_spectrogram/*'))
            df.to_csv(f'{main_path}/marking_spectrogram/batch_{count}.csv', index=False)


def learn_model():
    paths = glob(f'{main_path}/marking_spectrogram/*')
    features = []
    targets = []
    for path in paths:
        frame = pd.read_csv(path)
        features.append(np.array(frame.drop('label', axis=1)))
        targets.append(np.array(frame['label']))
    features = np.array(features)
    features = features.reshape(features.shape[0] * features.shape[1], 256, 256)
    targets = np.array(targets).reshape(-1)
    targets = to_categorical(targets)

    batch_size = 8
    consistent_features = consistent_return_segments(features, batch_size)
    consistent_targets = consistent_return_segments(targets, batch_size)

    print(consistent_features.shape)
    print(consistent_targets.shape)

    # double_consistent_features = np.append(consistent_features, consistent_features).reshape(-1, 256, 256)
    # double_consistent_targets = np.append(consistent_targets, consistent_targets).reshape(-1, 2)

    # sf.create_plots(double_consistent_features[0], '---', double_consistent_features[32:48],
    #                 double_consistent_targets[32:48])

    model = models.model_6(2, batch_size)
    model.fit(
        consistent_features,
        consistent_targets,
        shuffle=True,
        epochs=5,
        batch_size=8,
        steps_per_epoch=consistent_features.shape[0] // 8
    )
    model.save(f'{main_path}/models/model_5.h5')


def consistent_return_segments(segments, count_return_segments):
    start = 0
    stop = count_return_segments
    result = []
    while True:
        batch = segments[start:stop]
        result.append(batch)
        start += 1
        stop += 1
        if stop > len(segments):
            break
    return np.array(result)


def check_model():
    model = load_model(f'{main_path}/models/model_6_2.keras')
    paths = sf.get_bird_paths(0, 'train')
    np.random.shuffle(paths)
    for path in paths:
        audio, sr = lb.load(path)
        length = len(audio) / sr
        count_slices = int(length * Settings.count_slices_in_sec)
        main_spec = sf.spec_from_audio(audio, count_slices, 256)

        segments = sf.split_spectrogram(256, main_spec)
        sfp = sf.return_segments_for_plots(segments)
        for s in sfp:
            f, t = sf.get_batch_data(s, [1 for _ in range(len(s))], 3)
            prediction = model.predict(f)
            # prediction = np.argmax(prediction,axis=1)
            prediction = np.round(prediction, 2)
            print(prediction)
            sf.create_plots(main_spec, path, s[2:], prediction)


def preparation_data(batch_size):
    paths = glob(f'{main_path}/marking_spectrogram/*')
    features = []
    targets = []
    for path in paths:
        frame = pd.read_csv(path)
        features.append(np.array(frame.drop('label', axis=1)))
        targets.append(np.array(frame['label']))
    features = np.array(features)
    features = features.reshape(features.shape[0] * features.shape[1], 256, 256)
    targets = np.array(targets).reshape(-1)
    targets = to_categorical(targets)

    # print(r_features.shape)
    # print(r_targets.shape)
    #
    # for i in range(len(r_targets)):
    #     print(f'r_features: {r_targets[i]}, features: {targets[i+batch_size-1]}')
    #     fig,axes = plt.subplots(2,1)
    #     lb.display.specshow(r_features[i][-1],ax=axes[0])
    #     lb.display.specshow(features[i+batch_size-1],ax=axes[1])
    #     plt.show()

    return sf.get_batch_data(features, targets, batch_size)


def learn_model_2():
    batch_size = 3
    features, targets = preparation_data(3)
    print(features.shape, targets.shape)
    model = models.model_6(2, batch_size)

    early_stopping = EarlyStopping(
        monitor='accuracy',
        min_delta=0.01,
        verbose=1,
        patience=10,
        start_from_epoch=25,
        restore_best_weights=True
    )

    model.fit(
        features,
        targets,
        validation_split=0.1,
        epochs=50,
        batch_size=8,
        steps_per_epoch=features.shape[0] // 8,
        callbacks=[early_stopping]
    )
    model.save(f'{main_path}/models/model_6_2.keras')


def relearn_model():
    # - если прогнозы всех записей схожи между собой, то выбирается другая запись
    # - сохожесть вычисляется среднее по всем записям +-3 и все записи должны соответствовать
    # этому правилу
    # - порог правильности 0.8, если модель классифициирует все записи в правильные
    # или в неправильные, то запись также скипается
    # - обучение будет проходить по всем данным + по двум копированным новым. Одна эпоха
    def th(x, y):
        if y >= 0.9:
            return [0, 1]
        return [1, 0]

    def similarity_check(pre):
        print(pre)
        first_mean = np.mean([p[0] for p in pre])
        result = []
        for pr in pre:
            if (first_mean + 0.03 > pr[0]) and (first_mean - 0.03 < pr[0]):
                result.append('true')
        print(Counter(result)['true'])
        if Counter(result)['true'] > 12:
            return True
        return False

    break_ = False
    true_features, true_targets = preparation_data(3)
    model = load_model(f'{main_path}/models/model_6_2.keras')
    paths = sf.get_bird_paths(0, 'train')
    np.random.shuffle(paths)
    for path in paths:
        if len(target_paths) > 0:
            path = f'{main_path}/train_audio/asbfly/{target_paths.pop()}'
        audio, sr = lb.load(path)
        length = len(audio) / sr
        count_slices = int(length * Settings.count_slices_in_sec)
        main_spec = sf.spec_from_audio(audio, count_slices, 256)

        segments = sf.split_spectrogram(256, main_spec)
        sfp = sf.return_segments_for_plots(segments)
        for s in sfp:
            f, t = sf.get_batch_data(s, [1 for _ in range(len(s))], 3)

            prediction = model.predict(f)
            round_prediction = np.array([th(pre[0], pre[1]) for pre in prediction])
            similarity = similarity_check(prediction)
            round_similarity = similarity_check(round_prediction)
            if similarity or round_similarity:
                break_ = True
                break

            sf.create_plots(main_spec, path, s[2:], round_prediction)
            if input('True? >>>') != 't':
                continue

            features = np.append(true_features, f).reshape(-1, 3, 256, 256)
            targets = np.append(true_targets, round_prediction).reshape(-1, 2)

            model.fit(
                features, targets,
                batch_size=16,
                steps_per_epoch=f.shape[0] // 16,
                epochs=1
            )
        if break_:
            break_ = False
            continue
        model.save(f'{main_path}/models/model_6_2_retrain.keras')


relearn_model()
# научиться пользоваться генераторами
# реализовать все намеченные алгоритмы
input('><')
