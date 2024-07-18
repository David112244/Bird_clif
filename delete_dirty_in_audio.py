# хочу сделать модель которая будет обучаться превращать шум в пустоту
# и после этого подать на неё данные с шумом и событиями
# для этого нужно:
# 1. Взять случайную запись, сделать из неё спектрограмму
# 2. Найти фрагменты с шумом
# 3. Сделать модель которая будет превращать шум в пустоту
import pandas as pd
import numpy as np
import librosa as lb
from glob import glob
import matplotlib.pyplot as plt

import small_functions as sf

from PyEMD import EMD
from scipy.io.wavfile import write
import soundfile

main_path = 'E:/datas/birdclif'
get_species = lambda x: x.split('\\')[-1]
bird_species = [get_species(i) for i in glob(f'{main_path}/train_audio/*')]

species_path = np.array(glob(f'{main_path}/train_audio/{bird_species[0]}/*'))

path = species_path[10]
audio, sr = lb.load(path)
length = len(audio) / sr
count_slices = int(length * 128)
print(length)

emd = EMD()
mods = emd.emd(audio,max_imf=3)[:3]

spec_0 = sf.spec_from_audio(audio, count_slices, 256)
spec_1 = sf.spec_from_audio(mods[0], count_slices, 256)
spec_2 = sf.spec_from_audio(mods[1], count_slices, 256)
spec_3 = sf.spec_from_audio(mods[2], count_slices, 256)

print(spec_0.shape)

print(np.array(spec_0).shape)
print(np.array(spec_1).shape)
print(np.array(spec_2).shape)

for i in range(np.array(spec_0).shape[1] // 128):
    start_spec = i * 128
    finish_spec = (i + 1) * 128

    s_0 = spec_0[:, start_spec:finish_spec]
    s_1 = spec_1[:, start_spec:finish_spec]
    s_2 = spec_2[:, start_spec:finish_spec]
    s_3 = spec_3[:, start_spec:finish_spec]

    print(f's_0 max: {np.max(spec_0)}, mean: {np.mean(s_0)}, difference: {np.max(s_0)-np.min(s_0)}')
    print(f's_1 max: {np.max(spec_1)}, mean: {np.mean(s_1)}, difference: {np.max(s_1) - np.min(s_1)}')
    print(f's_2 max: {np.max(spec_2)}, mean: {np.mean(s_2)}, difference: {np.max(s_2) - np.min(s_2)}')
    print('\n')

    if np.max(s_0) > 0:
        fig, axes = plt.subplots(5, 1)
        lb.display.specshow(spec_0, ax=axes[0])
        lb.display.specshow(s_0, ax=axes[1])
        lb.display.specshow(s_1, ax=axes[2])
        lb.display.specshow(s_2, ax=axes[3])
        lb.display.specshow(s_3, ax=axes[4])
        plt.show()

# если 128 срезов это одна секунда, то если разлелить длину audio на количество срезав и умножить на 128 получиться одна секунда
# поэтому можно определять нахождение спектрограммы в аудио по индексу из спектрограммы
