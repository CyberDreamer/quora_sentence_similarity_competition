import numpy as np
import keras_file_manager
import pickle_file_manager
import datetime
import time
import sys
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import adagrad, adadelta
from keras.regularizers import l1, l2
# from keras import backend as K
# K.set_image_dim_ordering('th')

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

def indexToBinaryArray(index):
    res = np.zeros(5)
    temp = [int(x) for x in bin(index)[2:]]

    start = 5 - len(temp)
    res[start:] = temp

    return res

index_to_code = dict((i, indexToBinaryArray(i)) for i in xrange(28))

def TextToOneHotSymbolCode(data, lenght):
    ql = 200
    res_data = np.zeros((lenght, 2*ql, 5), dtype=np.int8)

    for j in range(lenght):
        res_row = res_data[j]
        q1 = data.iloc[j, 0]
        q2 = data.iloc[j, 1]

        for ii in range(ql):
            symbol = '_'
            if(ii<len(q1)):
                symbol = q1[ii]

            if(symbol in char_to_int):
                index = char_to_int[symbol]
                res_row[ii, :] = index_to_code[index]

            if (ii < len(q2)):
                symbol = q2[ii]

            if(symbol in char_to_int):
                index = char_to_int[symbol]
                res_row[ii + ql, :] = index_to_code[index]

        # if(j%1000==0):
        #     print(j)

    return res_data

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

            # print vector
        else:
            if(mode==0):
                res_row[i, :] = np.zeros(100)
            if(mode==1):
                res_row[i, :] = np.ones(100)

    # del sentence
    # sentence = None

    return res_row

def TextToVec(data, lenght):
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

def CreateModel(X, Y):
    img_rows, img_cols = X.shape[2], X.shape[3]

    model = Sequential()

    model.add(Convolution2D(28, 1, X.shape[3],
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    # activity_regularizer = activity_l1(0.001)
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(100, 2, 1))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 1)))

    # model.add(Convolution2D(256, 2, 1))
    # model.add(Activation('relu'))

    model.add(Convolution2D(300, 2, 1))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Convolution2D(600, 2, 1))
    model.add(Activation('relu'))

    # model.add(Convolution2D(2048, 2, 1))
    # model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(1, 1)))

    # model.add(Convolution2D(600, 2, 1))
    # model.add(Activation('relu'))

    model.add(Convolution2D(1200, 6, 1))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation('tanh'))

    return model

def GetModel(mode='create', filename='none', X=None, Y=None):

    model = None

    if (mode == 'create'):
        model = CreateModel(X=X, Y=Y)
        print("Neural net created...")

    if(mode=='load_W'):
        model = CreateModel(X=X, Y=Y)
        model.load_weights(filename)
        print("Neural net loaded...")

    if (mode == 'load_model'):
        model = keras_file_manager.LoadFromJSon(filename)
        print("Neural net loaded...")

    adag = adagrad()
    adad = adadelta()
    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model

def Get_Statisctic(data):
    a = data['question1'].str.len().values
    b = data['question2'].str.len().values
    sizes = np.concatenate((a,b), axis=0)

    print len(sizes)

    mean = np.mean(sizes)
    min = np.min(sizes)
    max = np.max(sizes)

    print 'MEAN words count: ', mean
    print 'MIN words count: ', min
    print 'MAX words count: ', max

    unique_sizes = np.bincount(sizes)
    ui = np.nonzero(unique_sizes)[0]
    res = zip(ui, unique_sizes[ui])
    print res
    exit()


def PrepareData(filename, start, end):
    data = pandas.read_csv(filename).iloc[start:end]
    data = textColumnsToLowcase(data)
    data = specSymbolReplacer(data)

    X = None
    Y = None

    if(filename=='train.csv'):
        data = data[['question1', 'question2', 'is_duplicate']].dropna()
        Y = data['is_duplicate'].values
        # Y = np_utils.to_categorical(Y, 2)
    if(filename=='test.csv'):
        data = data[['question1', 'question2']].fillna("Empty")

    X = data[['question1', 'question2']]
    print("Data loaded and prefiltered...")

    # Get_Statisctic(X)

    lenght = len(X)
    # X = TextToVec(X, lenght)
    X = TextToOneHotSymbolCode(X, lenght)

    print('samples count: ', X.shape[0])
    print('height: ', X.shape[1])
    print('matrix width: ', X.shape[2])
    # exit()

    X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
    del data
    print("Text to code translation finished...")

    return X, Y

# ----------------------- TRAIN  --------------------------
def Train(isNew, start, end):
    X, Y = PrepareData('train.csv', start, end)

    # pickle_file_manager.SaveToObject(X, 'binary_X_Train.dat')
    model = None
    if(isNew):
        model = GetModel(mode='create', filename=modelName, X=X, Y=Y)
    else:
        model = GetModel(mode='load_W', filename=weightsName, X=X, Y=Y)

    filepath = "w-imp-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X, Y, nb_epoch=epoch, batch_size=64, verbose=1, shuffle=True, callbacks=callbacks_list)

    keras_file_manager.SaveToJSon(model, modelName)
    print("Neural net saved...")
    # score = model.evaluate(trainX, y_train, verbose=1)
    # print "Model evaluated. Train score: ", score



def CrossTest(byWeights, start, end):
    X, Y = PrepareData('train.csv', start, end)

    model = None
    if(byWeights):
        model = GetModel(mode='load_W', filename=weightsName, X=X, Y=Y)
    else:
        model = GetModel(mode='load_model', filename=modelName, X=X, Y=Y)

    score = model.evaluate(X, Y, verbose=1, batch_size=32)
    # ans = model.predict(X, verbose=1, batch_size=32)
    #
    # lenght = end - start
    # for jk in xrange(lenght):
    #     print(ans[jk], '-', Y[jk])
    print(score)



# ----------------------- TEST --------------------------
def Test(byWeights):
    y_test = np.zeros(2345796)
    X, Y = PrepareData('test.csv', 0, 1000)
    model = None
    if(byWeights):
        model = GetModel(mode='load_W', filename=weightsName, X=X, Y=Y)
    else:
        model = GetModel(mode='load_model', filename=modelName, X=X, Y=Y)

    # step = 195483
    step = 390966
    # step = 390966
    ss = 6
    for m in range(ss):
        print("")
        start = step*m
        end = start + step

        X, Y = PrepareData('test.csv', start, end)
        predictions = model.predict(X, verbose=1, batch_size=64)

        y_test[start:end] = predictions[:,0]

        print((m+1), "/", ss, "  part predicted...")

        # ts = time.time()
        # st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        # print(st)

    print(y_test)

    pickle_file_manager.SaveToObject(y_test, 'y_result.csv')
    result = pandas.DataFrame(columns=['test_id'])
    result['is_duplicate'] = y_test

    resultName = 'result' + '_CNL+OneHot' + str(neurons) + '.csv'
    result.to_csv(resultName, index=False)
    print("Result saved...")




# -------------------------------------------------------
# ----------------------- MAIN --------------------------
# -------------------------------------------------------


question_lenght = 14
epoch = 40
neurons = (200, 300, 400, 500)
modelName = 'model' + '_CNL+OneHot' + str(neurons) + '.mdl'
weightsName = 'w-imp-08-0.2733.hdf5'


gc.enable()
# gc.disable()
# gc.set_threshold(100, 10, 10)
# lenght = 404290
# lenght = 2345796
Train(isNew=True, start=0, end=404290)
# CrossTest(byWeights=True, start=300000, end=404290)
Test(byWeights=False)