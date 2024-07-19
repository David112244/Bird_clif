import numpy as np
import pandas as pd

from tensorflow import keras
from keras import backend as K, Input, Model
from keras.layers import TimeDistributed, Concatenate, LSTM, GRU, Conv2D, Conv3D, Dense, Dropout, MaxPooling2D, \
    MaxPooling3D, Flatten, Add, \
    Concatenate, \
    GlobalMaxPooling2D
from keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

accuracy = Accuracy()
precision = Precision()
recall = Recall()


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    """
    Tversky loss function.
    Parameters:
    y_true (tensor): ground truth labels
    y_pred (tensor): predictions
    alpha (float): controls the penalty for false positives
    beta (float): controls the penalty for false negatives
    Returns:
    Tversky loss
    """
    true_positives = K.sum(y_true * y_pred)
    false_positives = K.sum((1 - y_true) * y_pred)
    false_negatives = K.sum(y_true * (1 - y_pred))
    tversky_index = (true_positives) / (true_positives + alpha * false_positives + beta * false_negatives)
    return 1 - tversky_index


def model1(out):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=[6, 6], activation='relu', input_shape=[64, 32, 1]))
    model.add(MaxPooling2D(pool_size=[3, 3]))

    model.add(Flatten())

    model.add(Dense(16, activation='relu'))
    model.add(Dense(out, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', metrics='accuracy', loss='categorical_crossentropy')  # окончательное создание сети

    return model


def model2(out):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=[10, 5], activation='relu', input_shape=[64, 128, 1]))
    model.add(Conv2D(16, kernel_size=[5, 5], activation='relu'))
    model.add(MaxPooling2D(pool_size=[2, 2]))

    model.add(Conv2D(16, kernel_size=[3, 2], activation='relu'))
    model.add(Conv2D(16, kernel_size=[2, 3], activation='relu'))
    model.add(MaxPooling2D(pool_size=[2, 2]))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(out, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', metrics='accuracy', loss='categorical_crossentropy')  # окончательное создание сети

    return model


def model_for_marking():
    model = Sequential()
    model.add(Conv2D(
        64, activation='relu',
        kernel_size=[50, 50],
        input_shape=(256, 256, 1),
        strides=[2, 2]
    ))
    model.add(MaxPooling2D(pool_size=[2, 2]))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=[3, 3],
                     activation='relu'
                     ))
    model.add(Conv2D(64, kernel_size=[3, 3],
                     activation='relu'
                     ))
    model.add(MaxPooling2D(pool_size=[2, 2]))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=[3, 3],
                     activation='relu'
                     ))
    model.add(MaxPooling2D(pool_size=[2, 2]))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer='adam', metrics='accuracy', loss='categorical_crossentropy')  # окончательное создание сети

    return model


# model = model_for_marking()
# model.fit()

def model_for_marking_2():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu', input_shape=(100, 100, 1)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(.1))

    model.add(Conv2D(32, kernel_size=[4, 4], activation='relu'))
    model.add(Conv2D(32, kernel_size=[3, 3], activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(.1))

    model.add(Conv2D(64, kernel_size=[3, 3], activation='relu'))
    model.add(Conv2D(64, kernel_size=[3, 3], activation='relu'))
    model.add(MaxPooling2D(pool_size=[3, 3]))
    model.add(Dropout(.1))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(1, activation='softmax'))
    model.summary()
    # model.compile(optimizer='adam', metrics='accuracy', loss='categorical_crossentropy')

    return model


def model3(count):
    model = Sequential()
    model.add(Conv2D(256, kernel_size=[3, 3], input_shape=[256, 256, count], activation='relu'))
    model.add(Conv2D(64, kernel_size=[3, 3], activation='relu'))
    model.add(MaxPooling2D(pool_size=[2, 2]))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, kernel_size=[3, 3], activation='relu'))
    model.add(Conv2D(64, kernel_size=[3, 3], activation='relu'))
    model.add(MaxPooling2D(pool_size=[2, 2]))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, kernel_size=[3, 3], activation='relu'))
    model.add(Conv2D(128, kernel_size=[3, 3], activation='relu'))
    model.add(MaxPooling2D(pool_size=[2, 2]))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, kernel_size=[3, 3], activation='relu'))
    model.add(Conv2D(64, kernel_size=[3, 3], activation='relu'))
    model.add(MaxPooling2D(pool_size=[2, 2]))
    model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(5, activation='sigmoid'))

    model.summary()
    model.compile(optimizer='adam', metrics=['binary_accuracy'],
                  loss='binary_crossentropy')  # окончательное создание сети

    return model


def model4(inp):
    model = Sequential()
    model.add(Conv2D(64, input_shape=[256, 256, inp], kernel_size=[5, 5], activation='relu'))
    model.add(Conv2D(64, kernel_size=[5, 5], activation='relu'))
    model.add(MaxPooling2D([3, 3]))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=[5, 5], activation='relu'))
    model.add(Conv2D(64, kernel_size=[5, 5], activation='relu'))
    model.add(MaxPooling2D([3, 3]))
    model.add(Dropout(0.2))

    # model.add(Conv2D(64, kernel_size=[5, 5], activation='relu'))
    # model.add(Conv2D(64, kernel_size=[5, 5], activation='relu'))
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=[5, 5], activation='relu'))
    model.add(Conv2D(64, kernel_size=[5, 5], activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer='adam', metrics='binary_accuracy',
                  loss='binary_crossentropy')

    return model


def model_5(out, batch_size):  # рекурентная
    model = Sequential()

    # Блок свёрточных слоёв 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(batch_size, 256, 256)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # Блок свёрточных слоёв 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # Блок свёрточных слоёв 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # Блок свёрточных слоёв 4
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # Рекуррентный блок
    model.add(TimeDistributed(Flatten()))
    model.add(GRU(128, return_sequences=True))
    model.add(GRU(128))

    # Полносвязные слои
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(out, activation='sigmoid'))

    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(),
                  metrics=['accuracy', precision, recall])
    model.summary()

    return model


def model_6(out, batch_size):  # рекурентная, способная обрабатывать последовательности
    model = Sequential()

    # Блок свёрточных слоёв 1
    model.add(Conv3D(32, (1, 3, 3), activation='relu', padding='same', input_shape=(None,batch_size, 256, 256,1)))
    model.add(Conv3D(32, (1, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((1,2, 2)))
    model.add(Dropout(0.2))

    # Блок свёрточных слоёв 2
    model.add(Conv3D(64, (1, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(64, (1, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((1,2, 2)))
    model.add(Dropout(0.2))

    # Блок свёрточных слоёв 3
    model.add(Conv3D(128, (2, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(128, (2, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((1,2, 2)))
    model.add(Dropout(0.2))

    # Блок свёрточных слоёв 4
    model.add(Conv3D(64, (1, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(64, (1, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((1,2, 2)))
    model.add(Dropout(0.2))

    # Рекуррентный блок
    model.add(TimeDistributed(Flatten()))
    model.add(GRU(128, return_sequences=True))
    model.add(GRU(256, return_sequences=True))
    model.add(GRU(256, return_sequences=True))
    model.add(GRU(256))

    # Полносвязные слои
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(out, activation='softmax'))

    model.compile(optimizer=Adam(0.0001), loss=CategoricalCrossentropy(),
                  metrics=['accuracy', precision, recall])
    model.summary()

    return model
