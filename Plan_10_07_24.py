import pandas as pd
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt

from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import models
import small_functions as sf
import medium_functions as mf

main_path = mf.main_path


def get_true_data():
    frame = pd.DataFrame()
    while 1:
        paths = glob(f'{main_path}/marking_spectrogram/*')
        index = np.random.randint(0, len(paths))
        path = paths[index]
        print(index)
        if path == r'E:/datas/birdclif/marking_spectrogram\used_paths.csv':
            continue
        break

    frame = pd.read_csv(path)

    columns_names = np.array(
        [
            [f'min_{i}' for i in range(3)],
            [f'max_{i}' for i in range(3)],
            [f'mean_{i}' for i in range(3)],
            [f'median_{i}' for i in range(3)],
            [f'average_{i}' for i in range(3)]
        ]).reshape(-1)
    columns_to_drop = np.concatenate([columns_names, ['Unnamed: 0']])
    # features = np.array(frame.drop(columns_to_drop, axis=1))
    # features = features.reshape(features.shape[0], 256, 256, 3)
    # targets = frame['label']
    return frame.drop(columns_to_drop, axis=1)


def init_model():
    model = models.model3(3)
    for i in range(10):
        frame = get_true_data()
        features = np.array(frame.drop('label', axis=1))
        targets = np.array(frame['label'])
        del frame
        features = features.reshape(features.shape[0], 256, 256, 3)
        x_train, x_val, y_train, y_val = train_test_split(features, targets)

        while 1:
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                batch_size=4,
                steps_per_epoch=x_train.shape[0] // 4,
                epochs=3
            )
            if history.history['binary_accuracy'][-1] > 0.6:
                break
            print('\nMore\n')

    model.save(f'{main_path}/models/model_2')


def relearn_model(count):
    def sure_threshold(x, threshold):
        if x > threshold:
            return 1
        return 0

    arr = []
    model = models.load_model(f'{main_path}/models/model_2')
    for _ in range(count):
        frame = get_true_data()
        true_features = np.array(frame.drop('label', axis=1))
        true_features = true_features.reshape(true_features.shape[0], 256, 256, 3)
        true_targets = np.array(frame['label'])
        print(true_features.shape)
        del frame

        validation_prediction = np.array([sure_threshold(i, 0.5) for i in model.predict(true_features)])
        acc = accuracy_score(true_targets, validation_prediction)
        arr.append(acc)
        print(arr)
        print(np.mean(arr))

        species_paths = sf.get_bird_paths(0, 'train')
        path = species_paths[np.random.randint(0, len(species_paths))]
        df = sf.get_split_mod_spectrogram(path, full=True)
        if type(df) != type(pd.DataFrame()):
            continue
        print(df.shape)

        features = np.array(df)
        del df
        features = features.reshape(features.shape[0], 256, 256, 3)
        prediction = np.array(model.predict(features)).reshape(-1)
        targets = np.array([sure_threshold(x, 0.5) for x in prediction])

        t = np.concatenate([targets, true_targets])
        f = np.concatenate([features, true_features])
        del features, targets, true_features, true_targets
        print(f'After: {f.shape}, {t.shape}')

        model.fit(
            f, t,
            steps_per_epoch=f.shape[0] // 4,
            batch_size=4,
            epochs=2
        )
    model.save(f'{main_path}/models/relearn_model_2')



relearn_model(100)

# объединить два фрейма (который сохнанён и которой создаётся) фрейм который сохнанён
# разделить на признаки и ответы. По сохранённому делать прогноз точности. Далее по двум
# фреймам дообучить модель
input('<>')
