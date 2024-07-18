import pandas as pd
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt

from glob import glob

import models
import small_functions as sf
import medium_functions as mf
from settings import Settings

from PyEMD import EMD
from sklearn.model_selection import train_test_split
from keras.models import load_model

main_path = mf.main_path


# предподготовка данный
def preparation_data(target_species_id):
    species_paths = np.array(sf.get_bird_paths(target_species_id, 'train'))
    paths_to_feature_id = sf.generate_random_numbers(5, len(species_paths))
    paths_to_feature = species_paths[paths_to_feature_id]

    frame = pd.DataFrame()
    for path in paths_to_feature:
        df = sf.get_split_mod_spectrogram(path)
        if type(df) != type(pd.DataFrame()):
            continue

        frame = pd.concat([frame, df], axis=0)
    frame['label'] = 1

    # frame.to_csv(f'{main_path}/labeled_spectrogram/frame.csv', index=False)
    random_species = mf.bird_species[sf.generate_random_numbers(20, len(mf.bird_species))]
    for i, species in enumerate(random_species):
        if i == target_species_id:
            continue
        while 1:
            print(i)
            paths = sf.get_bird_paths(i, 'train')
            path = paths[np.random.randint(0, len(paths))]
            df = sf.get_split_mod_spectrogram(path)
            if type(df) != type(pd.DataFrame()):
                continue
            df['label'] = 0
            frame = pd.concat([frame, df], axis=0)
            break
    print(frame.shape)
    return frame


# создаёт первый варианt новой модели
def learn_model(model, data, epochs):
    frame = data
    features = np.array(frame.drop('label', axis=1))
    features = features.reshape(features.shape[0], 256, 256, 1)
    targets = np.array(frame['label'])

    x_train, x_val, y_train, y_val = train_test_split(features, targets)

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=16,
        steps_per_epoch=x_train.shape[0] // 16,
        epochs=epochs
    )
    model.save(f'{main_path}/models/model_1')


def loop(circles, target_species_id, is_first):
    for _ in range(circles):
        data = preparation_data(target_species_id)
        if is_first:
            is_first = False
            model = models.model3(1)
            learn_model(model, data, 5)
        else:
            model = load_model(f'{main_path}/models/model_1')
            learn_model(model, data, 30)


def check_clef(target_species_id):
    model = load_model(f'{main_path}/models/model_1')
    species = mf.bird_species[target_species_id]
    species_path = glob(f'{main_path}/train_audio/{species}/*')[0]

    features = np.array(sf.get_split_mod_spectrogram(species_path, full_audio=True))
    features = features.reshape(features.shape[0], 256, 256, 1)
    prediction = model.predict(features)
    print(prediction)

    audio, sr = lb.load(species_path)
    length = len(audio) / sr
    count_slices = int(length * Settings.count_slices_in_sec)
    spec = sf.spec_from_audio(audio, count_slices, 256)
    lb.display.specshow(spec)
    plt.show()




# loop(10, 0, True)
check_clef(0)
