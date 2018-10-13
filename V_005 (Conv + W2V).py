import numpy as np
import keras_file_manager
import pickle_file_manager
import datetime
import time
import sys
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import adagrad
# from keras.regularizers import activity_l1, activity_l2

from gensim.models import word2vec
import gc




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


def SentenceToVec(sentence, mode):
    res_row = np.zeros((question_lenght, 100))
    sentence = sentence.split()

    lel = question_lenght
    if (len(sentence) < lel):
        lel = len(sentence)

    # start = offSet
    # end = start + 100
    for i, word in enumerate(sentence):
        if(i>=lel):
            break

        if (word in word2vec_model.wv):
            vector = word2vec_model.wv[word]
            res_row[i, :] = vector
        else:
            if(mode==0):
                res_row[i, :] = np.zeros(100)
            if(mode==1):
                res_row[i, :] = np.ones(100)

    # del sentence
    # sentence = None

    return res_row

def TextToOneHotCode(data, lenght):
    res_data = np.zeros((lenght, 2 * question_lenght, 100), dtype=np.float16)

    # print lenght
    # print len(data)
    # print 'TYPE: ', type(res_data)
    # exit()

    for index, row in data.iterrows():
        q1 = row['question1']
        q2 = row['question2']

    # for index in xrange(lenght):
    #     if index%2000==0:
    #         gc.collect()
    #
    #     q1 = data['question1'].iloc[index]
    #     q2 = data['question2'].iloc[index]

        if(index>=lenght):
            break

        res_data[index, 0:question_lenght, :] = SentenceToVec(q1, 0)
        res_data[index, question_lenght:2*question_lenght, :] = SentenceToVec(q2, 1)

        # print vectors
        # exit()

        # del q1
        # del q2
        # q1 = None
        # q2 = None

        # gc.collect()

    return res_data


def TextToVec(row, data):
    q1 = row['question1']
    q2 = row['question2']

    a = SentenceToVec(q1, 0)
    b = SentenceToVec(q2, 1)

    # row['code'] = np.concatenate((a,b), axis=1)
    c = np.concatenate((a, b), axis=1)
    data.append(c)

    del a
    del b
    del c
    a = None
    b = None
    c = None
    q1 = None
    q2 = None
    #
    # gc.collect()

    # print row['code']
    # print row['code'].shape
    # exit()

    # return row

def LaLa(dataFrame):
    q1_list = dataFrame['question1'].str.split().values.tolist()
    q2_list = dataFrame['question2'].str.split().values.tolist()

def CreateModel():
    nb_classes = 2

    img_rows, img_cols = 2*question_lenght, 100

    model = Sequential()

    model.add(Convolution2D(256, 1, 100,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    # activity_regularizer = activity_l1(0.001)
    # model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Convolution2D(128, 2, 1))
    # model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Convolution2D(128, 2, 1))
    # model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Convolution2D(32, 2, 1))
    # model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes))
    model.add(Activation('tanh'))

    # ada = adagrad()

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adagrad',
    #
    #               metrics=['accuracy'])

    return model


# ----------------------- TRAIN  --------------------------
def Train(epoch, modelName, start, end):
    train_data = pandas.read_csv('train.csv').iloc[start:end]
    train_data = textColumnsToLowcase(train_data)
    train_data = specSymbolReplacer(train_data)
    train_data = train_data[['question1', 'question2', 'is_duplicate']].dropna()
    x_train = train_data[['question1', 'question2']]
    y_train = train_data['is_duplicate'].values
    # x_train['code'] = pandas.Series(index=x_train.index)
    del train_data
    y_train = np_utils.to_categorical(y_train, 2)
    print("Train Data loaded and prefiltered...")
    lenght = len(x_train)

    x_train = TextToOneHotCode(x_train, lenght)
    # pickle_file_manager.SaveToObject(x_train,'train_set.dat')
    # exit()
    # data = []
    # x_train = x_train.apply(lambda cell: TextToVec(cell, data), axis=1)

    # x_train = x_train[['code']]
    # x_train = x_train[['code']].values[:][0]

    # print x_train.shape
    # x_train = x_train.apply(lambda cell: data.append(cell[['code']][:][0]), axis=1)
    # x_train = np.array(data)

    print("Text to code for train set finished...")

    model =  CreateModel()
    # filename = "weights-improvement-78-0.4877.hdf5"
    # model.load_weights(filename)

    # model = keras_file_manager.LoadFromJSon(modelName)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['accuracy'])
    print("Neural net loaded...")


    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    x_train = x_train.reshape((lenght, 1, 2*question_lenght, 100))
    model.fit(x_train, y_train, nb_epoch=epoch, batch_size=8, verbose=1, shuffle=True, callbacks=callbacks_list)


    keras_file_manager.SaveToJSon(model, modelName)
    print("Neural net saved...")
    # score = model.evaluate(trainX, y_train, verbose=1)
    # print "Model evaluated. Train score: ", score



def CrossTest(modelName, start, end):
    # model = CreateModel()
    # filename = "weights-improvement-78-0.4877.hdf5"
    # model.load_weights(filename)

    model = keras_file_manager.LoadFromJSon(modelName)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['accuracy'])
    print("Neural net loaded...")

    train_data = pandas.read_csv('train.csv').iloc[start:end]
    train_data = textColumnsToLowcase(train_data)
    train_data = specSymbolReplacer(train_data)
    train_data = train_data[['question1', 'question2', 'is_duplicate']].dropna()
    x_train = train_data[['question1', 'question2']]
    y_train = train_data['is_duplicate'].values
    # x_train['code'] = pandas.Series(index=x_train.index)

    y_train = np_utils.to_categorical(y_train, 2)

    print("Train Data loaded and prefiltered...")
    # lenght = end - start
    lenght = len(x_train)
    x_train = TextToOneHotCode(x_train, lenght)

    # data = []
    # x_train = x_train.apply(lambda cell: TextToVec(cell, data), axis=1)
    # x_train = np.array(data)
    print("Text to code for train set finished...")

    x_train = x_train.reshape((lenght, 1, 2*question_lenght, 100))
    score = model.evaluate(x_train, y_train, verbose=1, batch_size=8)
    # ans = model.predict(x_train, verbose=1, batch_size=8)

    # for jk in xrange(lenght):
    #     print ans[jk], '-', y_train[jk]
    print(score)



# ----------------------- TEST --------------------------
def Test(modelName):
    # model =  CreateModel()
    # filename = "weights-improvement-12-0.3852.hdf5"
    # model.load_weights(filename)

    model = keras_file_manager.LoadFromJSon(modelName)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    print("Neural net loaded...")
    y_test = np.zeros(2345796)
    # y_test = pickle_file_manager.LoadFromObject('step_12_temp_y_test.dat')

    source_data = pandas.read_csv('test.csv')
    source_data = textColumnsToLowcase(source_data)
    source_data = specSymbolReplacer(source_data)
    source_data = source_data[['question1', 'question2']].fillna("Empty")
    print("Data loaded and prefiltered...")

    # pickle_file_manager.SaveToObject(source_data, 'testSourceX.dat')
    # source_data = pickle_file_manager.LoadFromObject('testSourceX.dat')
    step = 195483
    # step = 1000
    # step = 390966
    ss = 12
    for m in range(ss):
        start = step*m
        end = start + step

        test_data = source_data.iloc[start:end]
        testX = TextToOneHotCode(test_data, step)
        print("Text to code for test set finished...")

        # testX_fileName = 'testX_part_' + str(m) + '.dat'
        # pickle_file_manager.SaveToObject(testX, testX_fileName)
        # testX = pickle_file_manager.LoadFromObject(testX_fileName)

        testX = testX.reshape((step, 1, 2*question_lenght, 100))
        predictions = model.predict(testX, verbose=1, batch_size=128)

        for jk in range(step):
            # print 'index', jk
            y_test[step*m + jk] = np.argmax(predictions[jk])
            # print predictions[jk]
            # print y_test[jk]

        filename = 'step_' + str(step) + ' _temp_y_test.dat'
        pickle_file_manager.SaveToObject(y_test, filename)

        print("")
        print((m+1), "/", ss, "  part predicted...")

        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        print(st)

    print(y_test)

    result = pandas.DataFrame(source_data.index, columns=['test_id'])
    result['is_duplicate'] = y_test

    resultName = 'result' + '_CNL+W2C' + str(neurons) + '.csv'
    result.to_csv(resultName, index=False)
    print("Result saved...")


question_lenght = 14
epoch = 100
neurons = (200, 300, 400, 500)
modelName = 'model' + '_CNL+W2C' + str(neurons) + '.mdl'

word2vec_model = word2vec.Word2Vec.load('word2vec_quora_set_model.mdl')
print('word2vec_model loaded')

gc.enable()
# gc.disable()
# gc.set_threshold(100, 10, 10)
# lenght = 404290
# lenght = 2345796
Train(epoch, modelName, 0, 254290)
CrossTest(modelName, 304290,404290)
# Test(modelName)