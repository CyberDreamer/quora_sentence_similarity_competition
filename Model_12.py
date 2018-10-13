from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import adagrad, adadelta
from keras import regularizers
from keras.layers import LSTM
from keras.regularizers import l2

class LSTM_M2:
    def __init__(self, N, X, Y):
        model = Sequential()
        model.add(LSTM(1000, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        model.add(LSTM(600, dropout_W=0.25))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        self.Model = model