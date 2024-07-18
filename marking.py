# как я буду осуществлять разметку:
# 1. Создать модель которая будет учавствовать в разметрке
# 2. Взять случайный путь к записи (случайного вида)
# 3. Сделать полную спектрограму записи, разбить запись на сегменты, выбрать 16 случайный
# сегментов, по ним сделать спектрограмы
# 4. Сделать прогноз модели по сегментам (есть голос птицы или нет)
# 5. Изобразить спектрограмы вместе с прогнозоми, самому оценить есть ли в спектрограме
# птичьий голос
# 6. Записать ответы, переобучить модель. Повторить всё начиная со второго пункта

# ! важно
import numpy as np
import pandas as pd
import librosa as lb
import matplotlib.pyplot as plt
from collections import Counter

from glob import glob
from time import sleep
from PyEMD import EMD
from time import time

# import models
import small_functions as sf
from settings import Settings

# import models
# from keras.models import load_model

main_path = 'E:/datas/birdclif'
get_species = lambda x: x.split('\\')[-1]
bird_species = [get_species(i) for i in glob(f'{main_path}/train_audio/*')]


# берёт по одному путю из папки каждого вида, преобразовывает в спектрограмы,
# сохраняет в raw_spec
def save_raw_spectrogram():
    sf.delete_all_files()
    paths = []
    for species in bird_species:
        species_paths = glob(f'{main_path}/train_audio/{species}/*')
        path = species_paths[np.random.randint(0, len(species_paths))]
        paths.append(path)
    for ind, path in enumerate(paths):
        print(f'{ind} of {len(paths)}')
        segments = sf.split_audio(path)
        file_name = path.split('\\')[-1].split('.')[0]
        for i, seg in enumerate(segments):
            spec = sf.spec_from_audio(seg, 256, 256)
            pd.DataFrame(spec).to_csv(f'{main_path}/raw_spectrogram/{file_name}_{i}.csv')


# выводит спектрограммы вместе с прогнозом модели потом запрашивает ответ
def display_of_spectrogram(bird_index):
    while 1:
        paths = np.array(glob(f'{main_path}/train_audio/'
                              f'{bird_species[bird_index]}/*'))
        path = paths[np.random.randint(0, len(paths))]
        print(f'\n{path}\n')
        audio, sr = lb.load(path)
        main_spec = sf.spec_from_audio(audio, hop=2048)

        segments = sf.split_audio(path)
        random_segments = segments[np.random.randint(0, len(segments), 16)]
        mini_specs = np.array([sf.spec_from_audio(seg, 100, 100) for seg in random_segments])
        print(mini_specs[0].shape)
        # predictions = cry_predict(mini_specs)
        # mini_id = [f'{i}__{p}' for i, p in enumerate(predictions)]
        mini_id = [f'{i}' for i in range(16)]
        create_plots(main_spec, path, mini_specs, mini_id)
        for _ in range(10):
            inp = input('>>>')
            if inp == 'stop':
                return
            elif inp == 'skip':
                break
            elif len(inp) == 16:
                answers = np.array([int(num) for num in inp])
                sf.save_frame_marking_spectrogram(mini_specs, answers, bird_species[bird_index])
                # relearn_model(mini_specs, answers)
                break
            else:
                print('Должно быть 16 ответов')


def create_plots(main_spec, main_name, mini_spec: np.array, mini_id):
    fig, axs = plt.subplots(ncols=4, nrows=5)
    gs = axs[0, 0].get_gridspec()
    fig.set_size_inches(12, 6)
    for ax in axs[0, 0:]:
        ax.remove()
    axbig = fig.add_subplot(gs[0, 0:])

    lb.display.specshow(main_spec, ax=axbig)
    axbig.set_title(main_name)
    index = 0
    for r in range(1, 5):
        for c in range(4):
            lb.display.specshow(mini_spec[index], ax=axs[r, c])
            axs[r, c].set_title(mini_id[index])
            index += 1
            if index >= len(mini_spec):
                fig.tight_layout()

                plt.show()
                return

    fig.tight_layout()

    plt.show()


# план разметки от 04.07
def marking_with_simple_model(ways):
    # used_paths = np.array(pd.read_csv(f'{main_path}/marking_spectrogram/used_paths.csv')).reshape(-1)
    # if len(ways) == len(used_paths):
    #     print('All records are marked')
    #     return
    while 1:
        path = ways[np.random.randint(0, len(ways))]

        audio, sr = lb.load(path)
        audio = sf.chop_audio(audio, sr,10)
        length = len(audio) / sr
        print(length)
        # if path in used_paths or length > 60:
        #     print('new')
        #     continue
        count_slices = int(length * 128)
        main_spec = sf.spec_from_audio(audio, count_slices, 256)

        lb.display.specshow(main_spec)
        plt.show()
        # emd = EMD()
        # mods = emd.emd(audio, max_imf=2)[:2]

        # path = np.array([path])
        # used_paths = np.concatenate([used_paths, path])
        # pd.DataFrame(used_paths).to_csv(f'{main_path}/marking_spectrogram/used_paths.csv', index=False)

        # mod0_spec = sf.spec_from_audio(mods[0], count_slices, 256)
        # mod1_spec = sf.spec_from_audio(mods[1], count_slices, 256)

        index = 0
        step = 256
        list_s_0 = sf.split_spectrogram(256, main_spec)
        # list_s_1 = sf.split_spectrogram(256, mod0_spec)
        # list_s_2 = sf.split_spectrogram(256, mod1_spec)

        print(path)

        start = 0
        stop = 16
        while 1:
            df_batch = pd.DataFrame()
            mini_spec_0 = np.array(list_s_0[start:stop])
            mini_spec_1 = np.array(list_s_1[start:stop])
            mini_spec_2 = np.array(list_s_2[start:stop])
            print(np.array(mini_spec_2).shape)
            create_plots(main_spec, path, mini_spec_0, [i for i in range(16)])
            start += 16
            stop += 16

            print(len(mini_spec_0))
            answers = sf.input_marking_answers(len(mini_spec_0))
            mins = np.round(
                [[np.min(j0), np.min(j1), np.min(j2)] for j0, j1, j2 in zip(mini_spec_0, mini_spec_1, mini_spec_2)], 2)
            maxs = np.round(
                [[np.max(j0), np.max(j1), np.max(j2)] for j0, j1, j2 in zip(mini_spec_0, mini_spec_1, mini_spec_2)], 2)
            means = np.round(
                [[np.mean(j0), np.mean(j1), np.mean(j2)] for j0, j1, j2 in zip(mini_spec_0, mini_spec_1, mini_spec_2)],
                2)
            medians = np.round([[np.median(j0), np.median(j1), np.median(j2)] for j0, j1, j2 in
                                zip(mini_spec_0, mini_spec_1, mini_spec_2)], 2)
            averages = maxs - mins

            array_data = np.hstack([mins, maxs, means, medians, averages])
            columns_names = np.array(
                [
                    [f'min_{i}' for i in range(3)],
                    [f'max_{i}' for i in range(3)],
                    [f'mean_{i}' for i in range(3)],
                    [f'median_{i}' for i in range(3)],
                    [f'average_{i}' for i in range(3)]
                ]).reshape(-1)
            data_df = pd.DataFrame(array_data, columns=columns_names)
            data_df['label'] = answers

            m1 = mini_spec_0.reshape(mini_spec_0.shape[0], -1)
            m2 = mini_spec_1.reshape(mini_spec_0.shape[0], -1)
            m3 = mini_spec_2.reshape(mini_spec_0.shape[0], -1)
            spec_data = np.concatenate([m1, m2, m3], axis=1)
            spec_df = pd.DataFrame(spec_data)
            df = pd.concat([spec_df, data_df], axis=1)
            count_batch = len(glob(f'{main_path}/marking_spectrogram/*')) - 1
            df.to_csv(f'{main_path}/marking_spectrogram/batch_{count_batch}.csv')

            if stop > len(list_s_0):
                break


# проверка правильности сохранения данных
def check_accuracy_save_data():
    df = pd.read_csv(f'{main_path}/marking_spectrogram/batch_1.csv').drop('Unnamed: 0', axis=1)
    line = df.iloc[15]
    specs = np.array(line[:256 * 256 * 3]).reshape(3, 256, 256)
    lb.display.specshow(specs[2])
    plt.show()

#
# marking_with_simple_model(sf.get_all_bird_paths('train'))
# input('><')
# добавить функцию что будет запоминать пути к аудио файлам и не выводить их снова
# создать модель которая по этим данным сможет определять есть на картинке событие или нет
