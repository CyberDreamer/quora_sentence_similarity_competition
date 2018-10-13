import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from scipy.sparse import hstack
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import Saver_Loader as save_loader

import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LSTM








alphabet = "_01 23abcdefghijklmnopqrstuvwxyz"
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

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

def StringToCode(string):
    result = ""
    iterator = 0

    for symbol in range(question_lenght):
        index = 0

        if(symbol in char_to_int):
            index = char_to_int[symbol]

        result += str(index)
        iterator+=1

    return result

def TextToCode(row):
    q1 = row['question1']
    # print len(q1)
    row['question1'] = StringToCode(q1)

    q2 = row['question2']
    # print len(q2)
    row['question2'] = StringToCode(q2)

    return row

def textColumnsToLowcase(data):
    data[['question1', 'question2']] = data[['question1', 'question2']]\
        .apply(lambda column: column.str.lower())

    return data

def specSymbolReplacer(data):
    data[['question1', 'question2']] = data[['question1', 'question2']].replace('[^a-zA-Z0-9]', ' ', regex=True)
    return data

def reagregate_data_1D(values):
    lenn = values.shape[0]
    # print lenn
    trainX = np.zeros((lenn, 2 * question_lenght, 1))
    for i in range(lenn):
        for j in range(question_lenght):
            trainX[i, j, 0] = values[i, 0][j]
            trainX[i, j + question_lenght, 0] = values[i, 1][j]

    return trainX

def reagregate_data_2D(values):
    lenn = values.shape[0]
    # print lenn
    trainX = np.zeros((lenn, question_lenght, 2))
    for i in range(lenn):
        for j in range(question_lenght):
            trainX[i, j, 0] = values[i, 0][j]
            trainX[i, j, 1] = values[i, 1][j]

    return trainX

def reagregate_data_DecWordD(values):
    lenn = values.shape[0]
    # print lenn
    dec_word_l = (2 * question_lenght)/delim
    half_dec_word_l = dec_word_l/2
    trainX = np.zeros((lenn, dec_word_l, delim))
    for i in range(lenn):
        for j in range(half_dec_word_l):
            for k in range(delim):
                index = k + j*delim
                trainX[i, j, k] = values[i, 0][index]
                trainX[i, j + half_dec_word_l, k] = values[i, 1][index]

    return trainX

def CreateModel():
    batch_size = 1
    dec_word_l = (2 * question_lenght)/delim

    model = Sequential()
    # model.add(LSTM(4, input_shape=(question_lenght, 1), return_sequences=True))
    model.add(LSTM(16, input_shape=(dec_word_l, delim)))
    # model.add(LSTM(32, batch_input_shape=(batch_size, question_lenght, 1), stateful=True, return_sequences=True))
    # model.add(LSTM(32, batch_input_shape=(batch_size, question_lenght, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model



# ----------------------- TRAIN  --------------------------
# ['test_id', 'question1', 'question2']
# .head(n=5000)
question_lenght = 150
lenght = 5000
delim = 15
# lenght = 404290
train_data = pandas.read_csv('train.csv').head(n=15000)

train_data = textColumnsToLowcase(train_data)

train_data = specSymbolReplacer(train_data)

train_data = train_data[['question1', 'question2', 'is_duplicate']].dropna()
x_train = train_data[['question1', 'question2']]
y_train = train_data['is_duplicate'].values

# x_train.apply(TextToOneHotCode, axis=1)

x_train.apply(TextToCode, axis=1)
print "Text to code for train set finished..."

# m1 = x_train['question1'].str.len().max()
# m2 = x_train['question2'].str.len().max()


trainX = reagregate_data_DecWordD(x_train.values)
# save_loader.SaveToObject(trainX, 'trainX_symb_code.mat')
print "reagregate data for train set finished..."


epoch = 25
neurons = (32,32)
model =  CreateModel()

print trainX.shape
epoch = 50
model.fit(trainX, y_train, nb_epoch=epoch, batch_size=1, verbose=1, shuffle=False)

# for i in range(epoch):
#     model.fit(trainX, y_train, nb_epoch=1, batch_size=1, verbose=2, shuffle=False)
#     model.reset_states()


modelName = 'model_E' + str(epoch) + '_LSTM' + str(neurons) + '.mdl'
save_loader.SaveToJSon(model, modelName)
score = model.evaluate(trainX, y_train, verbose=0)
print "train score", score



# ----------------------- TEST --------------------------
# model = save_loader.LoadFromJSon(modelName)
# lenght = 2345796
y_test = np.zeros(2345796)
ss = 4
for m in range(ss):
    start = 586449 * m
    end = start + 586449
    test_data = pandas.read_csv('test.csv').iloc[start:end]
    test_data = textColumnsToLowcase(test_data)
    test_data = specSymbolReplacer(test_data)
    test_data = test_data[['question1', 'question2']].fillna("Empty")

    test_data.apply(TextToCode, axis=1)
    print "Text to code for test set finished..."

    testX = reagregate_data_DecWordD(test_data.values)
    # save_loader.SaveToObject(testX, 'testX_symb_code.mat')
    print "reagregate data for test set finished..."

    y_test[start:end] = model.predict(testX)[0]
    print m, "/", ss, "  part predicted..."

print y_test

result = pandas.DataFrame(test_data.index, columns=['test_id'])
result['is_duplicate'] = y_test

resultName = 'result_E' + str(epoch) + '_LSTM' + str(neurons) + '.csv'
result.to_csv(resultName, index=False)
print "Result saved..."