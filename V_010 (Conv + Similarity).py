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
from keras import regularizers
# from keras import backend as K
# K.set_image_dim_ordering('th')

from gensim.models import word2vec
import gc
import os




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

    # model.add(Convolution2D(1280, 1, X.shape[3],
    model.add(Convolution2D(1024, 2, 2,
                            border_mode='valid',
                            input_shape=(2, img_rows, img_cols)))
                            # kernel_regularizer=regularizers.l2(0.01),
                            # activity_regularizer=regularizers.activity_l1(0.01)))
    # activity_regularizer = activity_l1(0.001)
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(1600, 3, 2))
                            # kernel_regularizer=regularizers.l2(0.01),
                            # activity_regularizer=regularizers.activity_l2(0.01)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(1, 2)))

    model.add(Convolution2D(2200, 1, 2))
                            # kernel_regularizer=regularizers.l2(0.01),
                            # activity_regularizer=regularizers.activity_l2(0.01)))
    model.add(Activation('relu'))

    # model.add(Convolution2D(800, 2, 2))
    # model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=(2, 1)))

    # model.add(Convolution2D(64, 2, 1,
    #                         activity_regularizer=regularizers.activity_l2(0.01)))
    # model.add(Activation('relu'))

    # model.add(Convolution2D(2048, 2, 1))
    # model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=(1, 1)))

    # model.add(Convolution2D(600, 2, 1))
    # model.add(Activation('relu'))

    # model.add(Convolution2D(1200, 6, 1))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(64))
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


def Get_word_statistics():
    sentences = pickle_file_manager.LoadFromObject('Quora_sentences.dat')
    print 'Data load...'
    lenght = len(sentences)
    print lenght
    sizes = np.zeros(lenght, dtype=np.int32)
    for ii in range(lenght):
        sizes[ii] = len(sentences[ii])

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

def Load_and_Prefiltred_Data(filename, start, end):
    data = pandas.read_csv(filename).iloc[start:end]
    data = textColumnsToLowcase(data)
    data = specSymbolReplacer(data)

    return data

def PrepareData(data, filename):
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
    X = SourceDataToSimilarityMap(X, lenght)

    print('samples count: ', X.shape[0])
    print('height: ', X.shape[2])
    print('matrix width: ', X.shape[3])
    # exit()

    # X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
    del data
    print("Text to code translation finished...")

    return X, Y

def create_directory(name):
    if not os.path.exists(name):
        os.makedirs(name)


def SourceDataToSimilarityMap(data, lenght):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    print(st)

    # fileName = experimentName + '/similarity_map_x_train' + '.csv'
    # result = pandas.read_csv(filename).values
    # print result

    import Translator_W2V_QuoraText_50
    result = Translator_W2V_QuoraText_50.TextToCode(data, 'W2V_Model_QuoraTT+Text8_50', lenght)

    # df = pandas.DataFrame()
    # df['is_duplicate'] = result
    # df.to_csv(fileName, index=False)
    # exit()

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    print(st)

    return result





# ----------------------- TRAIN  --------------------------
def Train(isNew, start, end):
    data = Load_and_Prefiltred_Data('train.csv', start, end)
    X, Y = PrepareData(data, 'train.csv')
    # pickle_file_manager.SaveToObject(X, 'similarityMap_TrainX')
    #
    # X = pickle_file_manager.LoadFromObject('similarityMap_TrainX')

    model = None
    if(isNew):
        model = GetModel(mode='create', filename=modelName, X=X, Y=Y)
    else:
        model = GetModel(mode='load_W', filename=weightsName, X=X, Y=Y)

    create_directory(experimentName)
    filepath = experimentName + "/w-imp-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X, Y, nb_epoch=epoch, batch_size=64, verbose=1, shuffle=True, callbacks=callbacks_list)

    keras_file_manager.SaveToJSon(model, modelName)
    print("Neural net saved...")
    # score = model.evaluate(trainX, y_train, verbose=1)
    # print "Model evaluated. Train score: ", score

def CrossTest(byWeights, start, end):
    data = Load_and_Prefiltred_Data('train.csv', start, end)
    X, Y = PrepareData(data, 'train.csv')
    # pickle_file_manager.SaveToObject(X, 'similarityMap_TrainX')

    # X = pickle_file_manager.LoadFromObject('similarityMap_TrainX')
    weightsName = GetLastCreatedFile(experimentName)
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

def SaveResult(Y, withIndex=True, ids=None):
    # pickle_file_manager.SaveToObject(y_test, 'y_result.csv')
    result = pandas.DataFrame()
    resultName = experimentName + '/result' + '.csv'

    if(withIndex==True):
        result['is_duplicate'] = Y
        result = result.reindex(result.index.rename('test_id'))
    else:
        result['test_id'] = ids
        result['is_duplicate'] = Y

    result.to_csv(resultName, index=withIndex)
    print("Result saved...")


def Test(byWeights):
    lenn = 2345796
    # y_test = np.zeros(lenn)

    fn = experimentName + '/part_y_test' + '.dat'
    y_test = pickle_file_manager.LoadFromObject(fn)
    data = Load_and_Prefiltred_Data('test.csv', 0, 100)
    X, Y = PrepareData(data, 'test.csv')

    model = None
    if(byWeights):
        model = GetModel(mode='load_W', filename=weightsName, X=X, Y=Y)
    else:
        model = GetModel(mode='load_model', filename=modelName, X=X, Y=Y)

    full_sourceData = Load_and_Prefiltred_Data('test.csv', 0, lenn)

    # step = 195483
    step = 390966
    ss = 6
    m = 1
    # for m in range(ss):
    print("")
    start = step*m
    end = start + step

    X, Y = PrepareData(full_sourceData[start:end], 'test.csv')
    predictions = model.predict(X, verbose=1, batch_size=64)

    y_test[start:end] = predictions[:,0]
    pickle_file_manager.SaveToObject(y_test, fn)

    print((m+1), "/", ss, "  part predicted...")

    # SaveResult(y_test, withIndex=True)



# -------------------------------------------------------
# ----------------------- MAIN --------------------------
# -------------------------------------------------------
def GetLastCreatedFile(directory):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]

    maxDate = None
    maxFile = ''
    for filename in onlyfiles:
        currentDate = os.stat(filename).st_mtime
        if(currentDate>maxDate):
            maxDate = currentDate
            maxFile = filename

    return maxFile


# Get_word_statistics()

question_lenght = 14
epoch = 25
neurons = (200, 300, 400, 500)
experimentName = 'CNL+Similar'
modelName =  experimentName + '/model' + '_CNL+Similar' + '.mdl'
weightsName = GetLastCreatedFile(experimentName)
# weightsName = experimentName + '/w-imp-01-0.5407.hdf5'

word2vec_model = word2vec.Word2Vec.load('word2vec_quora_set_model.mdl')
print('word2vec_model loaded')

gc.enable()
# gc.disable()
# gc.set_threshold(100, 10, 10)
# lenght = 404290
# lenght = 2345796
# Train(isNew=True, start=0, end=20000)
CrossTest(byWeights=True, start=200000, end=404290)
# Test(byWeights=True)


