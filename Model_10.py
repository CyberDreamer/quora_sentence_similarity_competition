from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import adagrad, adadelta
from keras import regularizers

class SquareConvNet_Adaptive:
    def __init__(self, N, X, Y):
        img_rows, img_cols = X.shape[2], X.shape[3]

        model = Sequential()
        map_count = 80
        model.add(Convolution2D(map_count * N, 3, 3,
                                border_mode='valid',
                                input_shape=(X.shape[1], img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        isFinished = False
        row_cn = 2
        col_cn = 2
        resedRows = X.shape[2]/row_cn
        resedCols = X.shape[3]/col_cn
        while(not isFinished):
            print row_cn, ' ', col_cn, ' ', map_count
            map_count=int(map_count*1.5)
            model.add(Convolution2D(map_count*N, row_cn, col_cn))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(1, 1)))


            resedRows = resedRows / row_cn
            resedCols = resedCols / col_cn

            if(resedRows>=1 and resedRows<=2):
                row_cn = 1

            if (resedCols >= 1 and resedCols <= 2):
                col_cn = 1

            if(row_cn==1 and col_cn==1):
                isFinished=True

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