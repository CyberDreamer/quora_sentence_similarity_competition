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




alphabet = "_01 23abcdefghijklmnopqrstuvwxyz"
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

def textColumnsToLowcase(data):
    data[['question1', 'question2']] = data[['question1', 'question2']]\
        .apply(lambda column: column.str.lower())

    return data

def specSymbolReplacer(data):
    data[['question1', 'question2']] = data[['question1', 'question2']].replace('[^a-zA-Z0-9]', ' ', regex=True)
    return data

def TextToOneHotCode(row):
    q1 = row['question1']
    q2 = row['question2']

    summary = q1 + " " + q2
    # print summary
    result = np.zeros((len(summary),len(alphabet)))

    iterator = 0
    for symbol in summary:
        if(symbol in char_to_int):
            index = char_to_int[symbol]
            code = np.zeros(len(alphabet))
            code[index] = 1
            result[iterator] = code
            iterator+=1

    # print result
    return result

def StringToCode(data):
    result = ""

    for ii in range(question_lenght):
        index = 0
        if(ii<len(data)):
            symbol = data[ii]
            if(symbol in char_to_int):
                index = char_to_int[symbol]

        result += str(index)

    return result

def TextToCode(row):
    q1 = row['question1']
    # print len(q1)
    row['question1'] = StringToCode(q1)

    q2 = row['question2']
    # print len(q2)
    row['question2'] = StringToCode(q2)

    return row

def reagregate_data(values):
    lenn = values.shape[0]
    # print lenn
    trainX = np.zeros((lenn, question_lenght, 2))
    for i in range(lenn):
        local_array = trainX[i]
        local_val_q1 = values[i, 0]
        local_val_q2 = values[i, 1]
        for j in range(question_lenght):
            local_array[j, 0] = local_val_q1[j]
            local_array[j, 1] = local_val_q2[j]
            # trainX[i, j, 0] = values[i, 0][j]
            # trainX[i, j, 1] = values[i, 1][j]

    return trainX


def CreateModel():
    batch_size = 128
    nb_classes = 2
    nb_epoch = 20

    # input image dimensions
    img_rows, img_cols = question_lenght, 2
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv_h = 2
    nb_conv_v = 5
    model = Sequential()

    model.add(Convolution2D(300, 2, 1,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(400, 2, 1))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(500, 2, 2))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(600, 2, 1))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(700, 2, 1))
    model.add(Activation('relu'))

    # model.add(Convolution2D(2048, 2, 1))
    # model.add(Activation('relu'))

    # model.add(Convolution2D(16, 2, 1))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filters, 2, 1))
    # model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model


# ----------------------- TRAIN  --------------------------
def Train(epoch, modelName, question_lenght):
    # ['test_id', 'question1', 'question2']
    # .head(n=5000)

    # lenght = 404290
    lenght = 300000
    # lenght = 404290
    train_data = pandas.read_csv('train.csv').head(n=lenght)
    train_data = textColumnsToLowcase(train_data)
    train_data = specSymbolReplacer(train_data)
    train_data = train_data[['question1', 'question2', 'is_duplicate']].dropna()
    x_train = train_data[['question1', 'question2']]
    y_train = train_data['is_duplicate'].values
    # print y_train[100:200]
    # print y_train[700:800]
    y_train = np_utils.to_categorical(y_train, 2)
    # print y_train[100:200]
    # print y_train[700:800]

    print "Train Data loaded and prefiltered..."

    x_train.apply(TextToCode, axis=1)
    print "Text to code for train set finished..."

    trainX = reagregate_data(x_train.values)
    print "reagregate data for train set finished..."
    trainX = trainX.reshape(trainX.shape[0], 1, question_lenght, 2)

    # save_loader.SaveToObject(trainX, 'trainX.dat')
    # save_loader.SaveToObject(y_train, 'y_train.dat')

    # trainX = save_loader.LoadFromObject('trainX.dat')
    # y_train = save_loader.LoadFromObject('y_train.dat')

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
    model.fit(trainX, y_train, nb_epoch=epoch, batch_size=128, verbose=1, shuffle=True, callbacks=callbacks_list)


    save_loader.SaveToJSon(model, modelName)
    print "Neural net saved..."
    # score = model.evaluate(trainX, y_train, verbose=1)
    # print "Model evaluated. Train score: ", score



# ----------------------- TEST --------------------------
def Test(modelName, question_lenght):
    # lenght = 2345796

    model =  CreateModel()
    filename = "weights-improvement-29-0.4354.hdf5"
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

    ss = 12
    for m in range(ss):
        start = 195483*m
        end = start + 195483

        test_data = source_data.iloc[start:end]
        test_data.apply(TextToCode, axis=1)
        print "Text to code for test set finished..."

        testX = reagregate_data(test_data.values)
        print "reagregate data for test set finished..."

        testX = testX.reshape(testX.shape[0], 1, question_lenght, 2)

        # testX_fileName = 'testX_part_' + str(m) + '.dat'
        # save_loader.SaveToObject(testX, testX_fileName)
        # testX = save_loader.LoadFromObject(testX_fileName)

        predictions = model.predict(testX, verbose=1, batch_size=20)[0]
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
epoch = 30
neurons = (128, 200, 400, 200, 128)
modelName = 'model' + '_CNL' + str(neurons) + '.mdl'

# Train(epoch, modelName, question_lenght)
Test(modelName, question_lenght)