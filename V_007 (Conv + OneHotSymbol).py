import numpy as np
import Saver_Loader as save_loader
import datetime
import time

import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils




alphabet = "_ abcdefghijklmnopqrstuvwxyz"
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

def textColumnsToLowcase(data):
    data[['question1', 'question2']] = data[['question1', 'question2']]\
        .apply(lambda column: column.str.lower())

    return data

def specSymbolReplacer(data):
    data[['question1', 'question2']] = data[['question1', 'question2']].replace('[^a-zA-Z0-9]', ' ', regex=True)
    return data

def TextToOneHotCode(data, lenght):
    res_data = np.zeros((lenght, question_lenght, 56))
    res_data = res_data.astype(int)

    for j in range(lenght):
        res_row = res_data[j]
        q1 = data.iloc[j, 0]
        q2 = data.iloc[j, 1]

        for ii in range(question_lenght):
            symbol_1 = '_'
            symbol_2 = '_'
            if(ii<len(q1)):
                symbol_1 = q1[ii]
            if (ii < len(q2)):
                symbol_2 = q2[ii]

            if(symbol_1 in char_to_int):
                index_1 = char_to_int[symbol_1]
                # print index_1
                res_row[ii, index_1] = 1

            if(symbol_2 in char_to_int):
                index_2 = char_to_int[symbol_2]
                res_row[ii, index_2 + 28] = 1

    return res_data


def CreateModel():
    batch_size = 128
    nb_classes = 2
    nb_epoch = 20

    # input image dimensions
    img_rows, img_cols = question_lenght, 56
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv_h = 2
    nb_conv_v = 5
    model = Sequential()

    model.add(Convolution2D(28, 1, 28,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(40, 2, 1))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(60, 2, 1))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(80, 2, 1))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(100, 2, 1))
    model.add(Activation('relu'))

    model.add(Convolution2D(120, 3, 2))
    model.add(Activation('relu'))

    # model.add(Convolution2D(400, 2, 2))
    # model.add(Activation('relu'))

    # model.add(Convolution2D(2048, 2, 1))
    # model.add(Activation('relu'))

    # model.add(Convolution2D(16, 2, 1))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filters, 2, 1))
    # model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model


# ----------------------- TRAIN  --------------------------
def Train(epoch, modelName, start, end):
    train_data = pandas.read_csv('train.csv').iloc[start:end]
    train_data = textColumnsToLowcase(train_data)
    train_data = specSymbolReplacer(train_data)
    train_data = train_data[['question1', 'question2', 'is_duplicate']].dropna()
    x_train = train_data[['question1', 'question2']]
    y_train = train_data['is_duplicate'].values

    y_train = np_utils.to_categorical(y_train, 2)
    print "Train Data loaded and prefiltered..."
    lenght = end - start
    trainX = TextToOneHotCode(x_train, lenght)
    print "Text to code for train set finished..."

    model =  CreateModel()
    # filename = "weights-improvement-04-2.6100.hdf5"
    # model.load_weights(filename)

    # model = save_loader.LoadFromJSon(modelName)

    # print "Neural net loaded..."
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    trainX = trainX.reshape((lenght, 1, question_lenght, 56))
    model.fit(trainX, y_train, nb_epoch=epoch, batch_size=32, verbose=1, shuffle=True, callbacks=callbacks_list)


    save_loader.SaveToJSon(model, modelName)
    print "Neural net saved..."
    # score = model.evaluate(trainX, y_train, verbose=1)
    # print "Model evaluated. Train score: ", score



def CrossTest(modelName, start, end):
    model = CreateModel()
    # filename = "weights-improvement-02-0.6434.hdf5"
    # model.load_weights(filename)

    # model = save_loader.LoadFromJSon(modelName)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adadelta',
    #               metrics=['accuracy'])
    print "Neural net loaded..."

    train_data = pandas.read_csv('train.csv').iloc[start:end]
    train_data = textColumnsToLowcase(train_data)
    train_data = specSymbolReplacer(train_data)
    train_data = train_data[['question1', 'question2', 'is_duplicate']].dropna()
    x_train = train_data[['question1', 'question2']]
    y_train = train_data['is_duplicate'].values

    y_train = np_utils.to_categorical(y_train, 2)
    print "Train Data loaded and prefiltered..."
    lenght = end - start
    trainX = TextToOneHotCode(x_train, lenght)
    print "Text to code for train set finished..."

    trainX = trainX.reshape((lenght, 1, question_lenght, 56))
    score = model.evaluate(trainX, y_train, verbose=1, batch_size=32)
    print score



# ----------------------- TEST --------------------------
def Test(modelName):
    model =  CreateModel()
    filename = "weights-improvement-02-0.6434.hdf5"
    model.load_weights(filename)

    # model = save_loader.LoadFromJSon(modelName)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adadelta',
    #               metrics=['accuracy'])
    print "Neural net loaded..."
    y_test = np.zeros(2345796)
    source_data = pandas.read_csv('test.csv')
    source_data = textColumnsToLowcase(source_data)
    source_data = specSymbolReplacer(source_data)
    source_data = source_data[['question1', 'question2']].fillna("Empty")
    print "Data loaded and prefiltered..."

    # save_loader.SaveToObject(source_data, 'testSourceX.dat')
    # source_data = save_loader.LoadFromObject('testSourceX.dat')
    lenght = 195483
    ss = 12
    for m in range(ss):
        start = 195483*m
        end = start + 195483

        test_data = source_data.iloc[start:end]
        testX = TextToOneHotCode(test_data)
        print "Text to code for test set finished..."

        # testX_fileName = 'testX_part_' + str(m) + '.dat'
        # save_loader.SaveToObject(testX, testX_fileName)
        # testX = save_loader.LoadFromObject(testX_fileName)

        testX = testX.reshape((lenght, 1, question_lenght, 56))
        predictions = model.predict(testX, verbose=1, batch_size=32)[0]
        print predictions
        y_test[start:end] = np.argmax(predictions)
        print y_test[start:end]
        print ""
        print (m+1), "/", ss, "  part predicted..."

        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        print st

    print y_test

    result = pandas.DataFrame(source_data.index, columns=['test_id'])
    result['is_duplicate'] = y_test

    resultName = 'result' + '_CNL' + str(neurons) + '.csv'
    result.to_csv(resultName, index=False)
    print "Result saved..."


question_lenght = 160
epoch = 20
neurons = (128, 200, 400, 200, 128)
modelName = 'model' + '_CNL' + str(neurons) + '.mdl'

# lenght = 404290
# lenght = 2345796
# Train(epoch, modelName, 0, 10000)
CrossTest(modelName, 50000,65000)
# Test(modelName)