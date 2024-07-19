import os

import numpy as np
import pandas as pd
from glob import glob
import librosa as lb
from collections import Counter
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from settings import Settings

main_path = Settings.main_path
get_species = lambda x: x.split('\\')[-1]
bird_species = [get_species(i) for i in glob(f'{main_path}/train_audio/*')]


def spec_from_audio(audio, hop=128, mel=128):
    sr = 22050
    m = np.nanmean(audio)
    audio = np.nan_to_num(audio, nan=m)
    audio[0] = 1

    raw_spec = lb.feature.melspectrogram(y=audio, sr=sr, hop_length=len(audio) // hop, n_mels=mel)

    width = (raw_spec.shape[1] // 32) * 32
    mel_spec_db = lb.power_to_db(raw_spec, ref=np.max)[:, :width]

    mel_spec_db = (mel_spec_db + 80) / 80  # -80~0db -> ~40~40db
    return mel_spec_db


def get_time(path):
    audio, sr = lb.load(path)
    return np.round(len(audio) / sr, 2)


def split_audio(path):
    audio, sr = lb.load(path)
    dur = sr * 4
    segments = []
    start = 0
    stop = dur
    while 1:
        seg = audio[int(start):int(stop)]
        segments.append(seg)
        start += dur / 8
        stop += dur / 8
        if stop > len(audio):
            break
    # lb.outputs.writeogg()
    return np.array(segments)


def split_spectrogram(len_step, spec):
    start = 0
    stop = len_step
    segments = []
    go = True
    while 1:
        seg = spec[:, int(start):int(stop)]
        segments.append(seg)
        start += len_step / 8
        stop += len_step / 8
        if not go:
            break
        if stop > spec.shape[1]:
            stop = spec.shape[1] - 1
            start = stop - len_step
            go = False
    return np.array(segments)


def count_each_species(sp=None):
    if not sp:
        count_for_every_species = []
        for path in glob(f'{main_path}/bird_spectrogram/*'):
            s = path.split('\\')[-1].split('_')[0]
            count_for_every_species.append(s)
        return Counter(count_for_every_species)
    else:
        count_species = []
        for path in glob(f'{main_path}/bird_spectrogram/*'):
            s = path.split('\\')[-1].split('_')[0]
            if s == sp:
                count_species.append(s)
        return Counter(count_species)


def delete_spectrogram(species, count):
    print(f'Delete: {species} - {count}')
    paths = glob(f'{main_path}/bird_spectrogram/*')
    species_path = []
    for path in paths:
        if path.split('\\')[-1].split('_')[0] == species:
            species_path.append(path)

    index = np.random.choice(range(len(species_path)), size=count, replace=False)
    for p in np.array(species_path)[index]:
        os.remove(p)


def delete_all_files():
    paths = glob(f'{main_path}/raw_spectrogram/*')
    for path in paths:
        os.remove(path)


def get_species(path):
    return path.split('\\')[-1].split('_')[0]


def get_index(species):
    di = {'asbfly': 0,
          'ashdro1': 1,
          'ashpri1': 2,
          'ashwoo2': 3,
          'asikoe2': 4,
          'asiope1': 5,
          'aspfly1': 6,
          'aspswi1': 7,
          'barfly1': 8,
          'barswa': 9,
          'bcnher': 10,
          'bkcbul1': 11,
          'bkrfla1': 12,
          'bkskit1': 13,
          'bkwsti': 14,
          'bladro1': 15,
          'blaeag1': 16,
          'blakit1': 17,
          'blhori1': 18,
          'blnmon1': 19,
          'blrwar1': 20,
          'bncwoo3': 21,
          'brakit1': 22,
          'brasta1': 23,
          'brcful1': 24,
          'brfowl1': 25,
          'brnhao1': 26,
          'brnshr': 27,
          'brodro1': 28,
          'brwjac1': 29,
          'brwowl1': 30,
          'btbeat1': 31,
          'bwfshr1': 32,
          'categr': 33,
          'chbeat1': 34,
          'cohcuc1': 35,
          'comfla1': 36,
          'comgre': 37,
          'comior1': 38,
          'comkin1': 39,
          'commoo3': 40,
          'commyn': 41,
          'compea': 42,
          'comros': 43,
          'comsan': 44,
          'comtai1': 45,
          'copbar1': 46,
          'crbsun2': 47,
          'cregos1': 48,
          'crfbar1': 49,
          'crseag1': 50,
          'dafbab1': 51,
          'darter2': 52,
          'eaywag1': 53,
          'emedov2': 54,
          'eucdov': 55,
          'eurbla2': 56,
          'eurcoo': 57,
          'forwag1': 58,
          'gargan': 59,
          'gloibi': 60,
          'goflea1': 61,
          'graher1': 62,
          'grbeat1': 63,
          'grecou1': 64,
          'greegr': 65,
          'grefla1': 66,
          'grehor1': 67,
          'grejun2': 68,
          'grenig1': 69,
          'grewar3': 70,
          'grnsan': 71,
          'grnwar1': 72,
          'grtdro1': 73,
          'gryfra': 74,
          'grynig2': 75,
          'grywag': 76,
          'gybpri1': 77,
          'gyhcaf1': 78,
          'heswoo1': 79,
          'hoopoe': 80,
          'houcro1': 81,
          'houspa': 82,
          'inbrob1': 83,
          'indpit1': 84,
          'indrob1': 85,
          'indrol2': 86,
          'indtit1': 87,
          'ingori1': 88,
          'inpher1': 89,
          'insbab1': 90,
          'insowl1': 91,
          'integr': 92,
          'isbduc1': 93,
          'jerbus2': 94,
          'junbab2': 95,
          'junmyn1': 96,
          'junowl1': 97,
          'kenplo1': 98,
          'kerlau2': 99,
          'labcro1': 100,
          'laudov1': 101,
          'lblwar1': 102,
          'lesyel1': 103,
          'lewduc1': 104,
          'lirplo': 105,
          'litegr': 106,
          'litgre1': 107,
          'litspi1': 108,
          'litswi1': 109,
          'lobsun2': 110,
          'maghor2': 111,
          'malpar1': 112,
          'maltro1': 113,
          'malwoo1': 114,
          'marsan': 115,
          'mawthr1': 116,
          'moipig1': 117,
          'nilfly2': 118,
          'niwpig1': 119,
          'nutman': 120,
          'orihob2': 121,
          'oripip1': 122,
          'pabflo1': 123,
          'paisto1': 124,
          'piebus1': 125,
          'piekin1': 126,
          'placuc3': 127,
          'plaflo1': 128,
          'plapri1': 129,
          'plhpar1': 130,
          'pomgrp2': 131,
          'purher1': 132,
          'pursun3': 133,
          'pursun4': 134,
          'purswa3': 135,
          'putbab1': 136,
          'redspu1': 137,
          'rerswa1': 138,
          'revbul': 139,
          'rewbul': 140,
          'rewlap1': 141,
          'rocpig': 142,
          'rorpar': 143,
          'rossta2': 144,
          'rufbab3': 145,
          'ruftre2': 146,
          'rufwoo2': 147,
          'rutfly6': 148,
          'sbeowl1': 149,
          'scamin3': 150,
          'shikra1': 151,
          'smamin1': 152,
          'sohmyn1': 153,
          'spepic1': 154,
          'spodov': 155,
          'spoowl1': 156,
          'sqtbul1': 157,
          'stbkin1': 158,
          'sttwoo1': 159,
          'thbwar1': 160,
          'tibfly3': 161,
          'tilwar1': 162,
          'vefnut1': 163,
          'vehpar1': 164,
          'wbbfly1': 165,
          'wemhar1': 166,
          'whbbul2': 167,
          'whbsho3': 168,
          'whbtre1': 169,
          'whbwag1': 170,
          'whbwat1': 171,
          'whbwoo2': 172,
          'whcbar1': 173,
          'whiter2': 174,
          'whrmun': 175,
          'whtkin2': 176,
          'woosan': 177,
          'wynlau1': 178,
          'yebbab1': 179,
          'yebbul3': 180,
          'zitcis1': 181}
    return di[species]


# функция расчитывающая сколоко каких классо и делящая их в заданной пропорции
def train_test_rate(rate=0.01):
    train_paths = []
    test_paths = []
    for path in glob(f'{main_path}/train_audio/*'):
        species = path.split('\\')[-1]
        paths = glob(f'{main_path}/train_audio/{species}/*')
        train, test = train_test_split(paths, test_size=rate, random_state=1)
        [train_paths.append(i) for i in train]
        [test_paths.append(j) for j in test]
    return (np.array(train_paths), np.array(test_paths))


def marking(data):
    result = []
    for species, count in Counter(data).items():
        for _ in range(count):
            result.append(get_index(species))
    return np.array(result)


def more_or_less_05(num):
    if num > 0.5:
        return 1
    else:
        return 0


# создаёт фрэйм из размечнных спектрограм
def save_frame_marking_spectrogram(specs, answers, species):
    specs_array = np.array([spec.reshape(-1) for spec in specs])
    df = pd.DataFrame(specs_array)
    df['label'] = answers
    count_in_file = len(glob(f'{main_path}/marking_spectrogram/*'))
    df.to_csv(f'{main_path}/marking_spectrogram/batch_{species}_{count_in_file}.csv')


def generate_random_numbers(how_much, count):
    n = np.arange(how_much - 1)
    np.random.shuffle(n)
    return n[:count]


def input_marking_answers(count):
    if Settings.debug:
        return np.array([1 for _ in range(count)])
    else:
        inp = input(f'{count} ответов>>>')
        if inp == 'stop':
            return inp
        try:
            inp = [int(i) for i in inp]
        except ValueError:
            print('Ответы должны быть цифрами')
            inp = input_marking_answers(count)
        if len(inp) != count:
            print(f'Должно быть {count} ответов')
            inp = input_marking_answers(count)
        return np.array(inp)


# делит список из путей на тренировочные и тестовые пути
def train_test_paths(paths):
    train_paths, test_paths = train_test_split(paths, random_state=1, test_size=0.05)
    return [train_paths, test_paths]


# выводит тренировочные или тестовые пути определённого вида птицы
def get_bird_paths(bird_index, train_test):
    species = bird_species[bird_index]
    paths = glob(f'{main_path}/train_audio/{species}/*')
    paths = train_test_paths(paths)
    if train_test == 'train':
        return paths[0]
    elif train_test == 'test':
        return paths[1]


# возвращает тренировачные или тестовые пути всех видов птиц
def get_all_bird_paths(train_test):
    arr = []
    for i in range(len(bird_species)):
        for path in get_bird_paths(i, train_test):
            arr.append(path)
    return np.array(arr).reshape(-1)


# получает путь к файлу, выводит массив спектрограмм (одной натуральной и двух сделанный из
# эмпирических мод) разбитых по две секунды
def get_split_mod_spectrogram(path, l=10, full=False, full_audio=False):
    audio, sr = lb.load(path)
    if not full_audio:
        audio = chop_audio(audio, sr, output_length=10)

    length = len(audio) / sr
    count_slices = int(length * Settings.count_slices_in_sec)
    print(length)
    if (length > l and Settings.debug) or length < 2.2:
        print('skip')
        return None

    if full:
        mods = EMD().emd(audio, max_imf=2)[:2]

        main_spec = np.array(spec_from_audio(audio, count_slices, 256)).astype(np.float16)
        mod0_spec = np.array(spec_from_audio(mods[0], count_slices, 256)).astype(np.float16)
        mod1_spec = np.array(spec_from_audio(mods[1], count_slices, 256)).astype(np.float16)

        step = Settings.count_slices_in_step
        list_s_0 = split_spectrogram(step, main_spec)
        list_s_1 = split_spectrogram(step, mod0_spec)
        list_s_2 = split_spectrogram(step, mod1_spec)
    else:
        main_spec = np.array(spec_from_audio(audio, count_slices, 256)).astype(np.float16)
        step = Settings.count_slices_in_step
        list_s_0 = split_spectrogram(step, main_spec)
    arr = []
    if full:
        for item_s in zip(list_s_0, list_s_1, list_s_2):
            item_s = np.array(item_s).reshape(-1)
            arr.append(item_s)
        df = pd.DataFrame(np.array(arr))
    else:
        for item_s in list_s_0:
            item_s = np.array(item_s).reshape(-1)
            arr.append(item_s)
        df = pd.DataFrame(np.array(arr))

    return df


# вернуть из аудио рандомные 10 секунд
def chop_audio(audio, sr, output_length=5):
    length = len(audio) / sr
    if length < output_length:
        return audio
    min_start = 0
    max_start = len(audio) - output_length * sr

    start = np.random.randint(min_start, max_start)
    finish = start + output_length * sr
    result = audio[start:finish]
    return result


def get_digital_data_from_spec(spec):
    min_ = np.min(spec)
    max_ = np.max(spec)
    mean_ = np.mean(spec)
    median_ = np.median(spec)
    average_ = max_ - min_
    return np.array([min_, max_, mean_, median_, average_])


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


def return_segments_for_plots(segs):
    start = 0
    stop = 16
    result = []
    while 1:
        if stop > len(segs):
            batch = segs[-16:]
            result.append(batch)
            break
        batch = segs[start:stop]
        result.append(batch)
        start += 16
        stop += 16
    return np.array(result)

# принимает данные возврщает подвыборки размером batch_size сохраняя последовательность
def get_batch_data(features,targets, batch_size):
    r_features = []
    r_targets = []
    start = 0
    end = batch_size
    while True:
        if end > len(targets):
            break
        r_features.append(features[start:end])
        r_targets.append(targets[end - 1])
        start += 1
        end += 1
    r_features = np.array(r_features)
    r_targets = np.array(r_targets)
    return [r_features,r_targets]