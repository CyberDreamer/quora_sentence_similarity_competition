from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import adagrad, adadelta
from keras import regularizers

class HorizontalConvNetM1:
    def __init__(self, N, X, Y):
        img_rows, img_cols = X.shape[2], X.shape[3]

        model = Sequential()

        model.add(Convolution2D(100*N, 1, X.shape[3],
                                border_mode='valid',
                                input_shape=(2, img_rows, img_cols)))
        model.add(Activation('relu'))

        # model.add(MaxPooling2D(pool_size=(2, 1)))

        model.add(Convolution2D(80*N, 2, 1))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 1)))

        model.add(Convolution2D(60*N, 3, 1))
        model.add(Activation('relu'))

        # model.add(Convolution2D(800, 2, 2))
        # model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(1, 1)))

        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1))
        model.add(Activation('tanh'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        self.Model = model