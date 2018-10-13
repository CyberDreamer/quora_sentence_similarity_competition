from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import adagrad, adadelta
from keras import regularizers
from keras.layers import LSTM
from keras.applications import ResNet50

class LSTM_M1:
    def __init__(self, N, X, Y):
        model = Sequential()
        model.add(LSTM(N, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        # model.add(LSTM(800, return_sequences=True))
        model.add(LSTM(N))
        # model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        self.Model = model