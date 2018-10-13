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
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils import np_utils
from keras.optimizers import adagrad, adadelta
from keras import regularizers
from MyEarlyStoppingKeras import EarlyStoppingByLossVal
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
    # data = textColumnsToLowcase(data)
    # data = specSymbolReplacer(data)

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

    try:
        print('samples count: ', X.shape[0])
        print('height: ', X.shape[2])
        print('matrix width: ', X.shape[3])
    except: IndexError

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
    # import Translator_W2V_QuoraText_50
    # result = Translator_W2V_QuoraText_50.TextToCode(data, 'W2V_Model_QuoraTT+Text8_50', lenght)

    # import Translator_SimMatrix
    # result = Translator_SimMatrix.TextToCode(data, 'W2V_Model_QuoraTT+Text8_50', lenght)

    # import Translator_LSTM
    # result = Translator_LSTM.TextToCode(data, 'W2V_Model_QuoraTT_50', lenght, maxQLen=20)
    # result = Translator_LSTM.TextToCodeNorm(data, 'W2V_Model_QuoraTT_50', lenght, maxQLen=20)
    # result = Translator_LSTM.TextToCodeByList(data, 'W2V_Model_QuoraTT_50', lenght, maxQLen=20)

    import Translator_LSTM_Summary
    result = Translator_LSTM_Summary.TextToSCodeNorm(data, 'W2V_Model_QuoraTT_50', lenght, maxQLen=5)


    # df = pandas.DataFrame()
    # df['is_duplicate'] = result
    # df.to_csv(fileName, index=False)
    # exit()

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    print(st)

    return result





# ----------------------- TRAIN  --------------------------
def Train(model, experimentName, X, Y):
    create_directory(experimentName)
    filepath = experimentName + "/w-imp-{epoch:02d}-{loss:.4f}.hdf5"
    filename = experimentName + "/train_history.csv"
    logger = CSVLogger(filename, separator=',', append=True)
    # stopper = EarlyStoppingByLossVal(monitor='val_loss', value=0.001, verbose=0)
    stopper = EarlyStopping(monitor='loss', min_delta=0.0, patience=3, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint, logger, stopper]

    model.fit(X, Y, nb_epoch=epoch, batch_size=64, verbose=1, shuffle=True, callbacks=callbacks_list)

    modelName = experimentName + "/CN_weights.mdl"
    model.save_weights(modelName)
    # keras_file_manager.SaveToJSon(model, modelName)
    print("Neural net saved...")
    # score = model.evaluate(trainX, y_train, verbose=1)
    # print "Model evaluated. Train score: ", score

def CrossTest(model, X, Y):
    score = model.evaluate(X, Y, verbose=1, batch_size=32)
    # ans = model.predict(X, verbose=1, batch_size=32)
    #
    # lenght = end - start
    # for jk in xrange(lenght):
    #     print(ans[jk], '-', Y[jk])
    # print(score)
    return score


# ----------------------- TEST --------------------------

def SaveResult(Y, experimentName, withIndex=True, ids=None):
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


def Test(model, experimentName):
    lenn = 2345796
    y_test = np.zeros(lenn)

    fn = experimentName + '/part_y_test' + '.dat'
    # y_test = pickle_file_manager.LoadFromObject(fn)
    # data = Load_and_Prefiltred_Data('test.csv', 0, 100)
    # X, Y = PrepareData(data, 'test.csv')

    full_sourceData = Load_and_Prefiltred_Data('test.csv', 0, lenn)

    step = 195483
    # step = 10000
    # step = 390966
    ss = 12
    for m in range(0, ss, 1):
        print("")
        start = step*m
        end = start + step

        X, Y = PrepareData(full_sourceData[start:end], 'test.csv')
        predictions = model.predict(X, verbose=1, batch_size=128)

        y_test[start:end] = predictions[:,0]
        # pickle_file_manager.SaveToObject(y_test, fn)

        print((m+1), "/", ss, "  part predicted...")

    SaveResult(y_test, experimentName, withIndex=True)



def ShiftResultToNormal(filename):
    data = pandas.read_csv(filename)
    data[['is_duplicate']] = data[['is_duplicate']].apply(lambda row: (row + 1) / 2)
    data.to_csv(filename, index=False)
    exit()

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


gc.enable()
# word2vec_model = word2vec.Word2Vec.load('word2vec_quora_set_model.mdl')
# print('word2vec_model loaded')


epoch = 15

# delta = 20000
data = Load_and_Prefiltred_Data('train.csv', 0, 200000)
X_1_Train, Y_1_Train = PrepareData(data[0:200000], 'train.csv')
# X_1_Test, Y_1_Test = PrepareData(data[300000:404290], 'train.csv')

# X_2_Train, Y_2_Train = PrepareData(data[80000:100000], 'train.csv')
# X_2_Test, Y_2_Test = PrepareData(data[100000:160000], 'train.csv')

from Model_1 import HorizontalConvNetM1
from Model_2 import SquareConvNet
from Model_3 import HorizontalConvNetM3
from Model_4 import SquareConvNet_M2
from Model_5 import SquareConvNet_M3
from Model_6 import HorizontalConvNetM4
from Model_7 import HorizontalConvNetM5
from Model_8 import HorizontalConvNetM6
from Model_9 import SquareConvNet_M4
from Model_10 import SquareConvNet_Adaptive
from Model_11 import LSTM_M1
from Model_12 import LSTM_M2
from Model_13 import LSTM_M3

from sklearn import ensemble

# ens_model = ensemble.
# ens_model.fit(X_1_Train, Y_1_Train)
# score = ens_model.score(X_1_Test, Y_1_Test)
# print score
# exit()

# Exp_1 = ('SquareConvNet_M4 5 + Similar', SquareConvNet_M4(5, X_1_Train, Y_1_Train))
# Exp_2 = ('SquareConvNetAdap 5 + SimilarMatrix', SquareConvNet_Adaptive(5, X_1_Train, Y_1_Train))



# M = LSTM_M1(800, X_1_Train, Y_1_Train)
# expName = 'LSTM_M1 800 Full W2V'
# M.Model.load_weights(expName + '/w-imp-03-0.0858.hdf5')
# score = CrossTest(M.Model, X_1_Test, Y_1_Test)
# print score
# exit()

# Test(M.Model, expName)
# exit()


# Exp_1 = ('LSTM_M1 800 Full W2V', M)

# M = LSTM_M2(1000, X_1_Train, Y_1_Train)
# expName = 'LSTM_M2 1000 W2V'
# M.Model.load_weights(expName + '/w-imp-03-0.1212.hdf5')
# Test(M.Model, expName)
# exit()

M = LSTM_M1(128, X_1_Train, Y_1_Train)
expName = 'LSTM_M1 128 W2V'
# M.Model.load_weights(expName + '/w-imp-26-0.0708.hdf5')
# Test(M.Model, expName)
# exit()

Exp_2 = (expName, M)

Experiments = [Exp_2]

for e in Experiments:
    expName, model = e
    Train(model.Model, expName, X_1_Train, Y_1_Train)
    # score = CrossTest(model.Model, X_1_Test, Y_1_Test)
    # print '------------------------------------------'
    # print ''
    # print 'First Stage Experiment on ', expName, ' Score = ', score
    # print ''
    # print '------------------------------------------'


    # Train(model, expName, X_2_Train, Y_2_Train)
    # score = CCrossTest(model, X_2_Test, Y_2_Test)
    # print 'Second Stage Experiment on ', expName, ' Score = ', score

print 'Exprement FINISHED!'
