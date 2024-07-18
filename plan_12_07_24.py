import pandas as pd
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt

from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import small_functions as sf
import medium_functions as mf
from settings import Settings
from marking import create_plots

from joblib import dump, load

main_path = mf.main_path
# columns_names = np.array(
#     [
#         [f'min_{i}' for i in range(3)],
#         [f'max_{i}' for i in range(3)],
#         [f'mean_{i}' for i in range(3)],
#         [f'median_{i}' for i in range(3)],
#         [f'average_{i}' for i in range(3)],
#
#     ]).reshape(-1)

columns_names = np.array(['min_0', 'max_0', 'mean_0', 'median_0', 'average_0'])


def save_digital_features():
    global columns_names
    columns_names = np.append(columns_names, ['label'])
    frame = pd.DataFrame()
    for path in glob(f'{main_path}/marking_spectrogram/*'):
        if path == r'E:/datas/birdclif/marking_spectrogram\used_paths.csv':
            continue
        df = pd.read_csv(path)
        needful_df = df[columns_names]
        print(needful_df.columns)
        frame = pd.concat([frame.reset_index(drop=True), needful_df.reset_index(drop=True)], axis=0)
    frame.to_csv(f'{main_path}/marking_spectrogram/digital_features_frame.csv', index=False)


def lear_model():
    while 1:
        frame = pd.read_csv(f'{main_path}/marking_spectrogram/digital_features_frame.csv')
        features = frame.drop('label', axis=1)
        print(features.shape)
        targets = frame['label']
        x_train, x_test, y_train, y_test = train_test_split(features, targets)

        forest = RandomForestClassifier(n_jobs=-1)
        forest.fit(x_train, y_train)
        test_score = forest.score(x_test, y_test)
        train_score = forest.score(x_train, y_train)
        print(f'Test: {test_score}\nTrain: {train_score}\n')
        if test_score > 0.99:
            dump(forest, f'{main_path}/models/marking_forest')
            return


def visual_check_accuracy():
    global columns_names
    paths = sf.get_all_bird_paths('train')
    forest = load(f'{main_path}/models/marking_forest')
    for path in paths:
        df = sf.get_split_mod_spectrogram(path, full=True)
        arr = []
        for i in range(df.shape[0]):
            row = np.array(df.iloc[i])
            specs = row.reshape(3, 256, 256)

            # fig, axes = plt.subplots(3, 1)
            # lb.display.specshow(specs[0],ax=axes[0])
            # lb.display.specshow(specs[1],ax=axes[1])
            # lb.display.specshow(specs[2],ax=axes[2])
            # plt.show()

            mins = np.round([np.min(j) for j in specs], 2)
            maxs = np.round([np.max(j) for j in specs], 2)
            means = np.round([np.mean(j) for j in specs], 2)
            medians = np.round([np.median(j) for j in specs], 2)
            averages = maxs - mins

            array_data = np.hstack([mins, maxs, means, medians, averages])
            arr.append(array_data)
        arr = np.array(arr)
        predictions = forest.predict(arr)

        audio, sr = lb.load(path)
        length = len(audio) / sr
        count_slices = int(length * Settings.count_slices_in_sec)
        main_spec = sf.spec_from_audio(audio, count_slices, 256)
        mini_specs = np.array(df.iloc[:16])
        mini_specs = mini_specs.reshape(16, 3, 256, 256)[:, 0:1, :, :].reshape(16, 256, 256)

        create_plots(main_spec, path, mini_specs, predictions)


lear_model()
visual_check_accuracy()
