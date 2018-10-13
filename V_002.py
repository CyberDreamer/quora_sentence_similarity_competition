import numpy as np
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from scipy import sparse
import Saver_Loader as save_loader

import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D




def textColumnsToLowcase(data):
    data[['question1', 'question2']] = data[['question1', 'question2']]\
        .apply(lambda column: column.str.lower())

    return data

def specSymbolReplacer(data):
    data[['question1', 'question2']] = data[['question1', 'question2']].replace('[^a-zA-Z0-9]', ' ', regex=True)
    return data

def SourceTextToMatrix(corpus):
    # sentences = word2vec.Text8Corpus(corpus)
    model = word2vec.Word2Vec(corpus, workers=4, size=100, min_count=50, window=10, sample=1e-3)
    model.save('word2vec_data.mdl')
    ans = model.wv['what']
    print ans
    word2vec.
    # model = Word2Vec.load('word2vec_data.mdl')

def CreateModel():
    model = Sequential()

    model.add(Dense(20, input_dim=11724))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(loss='mse',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model


def getUniqueWordsFromPart(part):
    words_split = part.str.split(' ', expand=True)
    words_unique = words_split.stack().unique()
    words = list(words_unique)

    return words

# ['test_id', 'question1', 'question2']
train_data = pandas.read_csv('train.csv').head(n=5000)
test_data = pandas.read_csv('test.csv').head(n=5000)


train_data = textColumnsToLowcase(train_data)
test_data = textColumnsToLowcase(test_data)

train_data = specSymbolReplacer(train_data)
test_data = specSymbolReplacer(test_data)

vectorizer = TfidfVectorizer()
train_data = train_data[['question1', 'question2', 'is_duplicate']].dropna()
x_train = train_data[['question1', 'question2']]

words_1 = getUniqueWordsFromPart(x_train['question1'])
words_2 = getUniqueWordsFromPart(x_train['question2'])

total_words = list(set(words_1) | set(words_2))

# print total_words
# exit()
y_train = train_data['is_duplicate'].values

test_data = test_data[['question1', 'question2']].fillna("Empty")


SourceTextToMatrix(total_words)
exit()





x_train = train_q1_m + train_q2_m
x_test = test_q1_m + test_q2_m

# x_train = x_train.toarray()
# x_test = x_test.toarray()
# print x_train
# print "max features: ", len(vectorizer.vocabulary_)
# print "sparse matrix shape: ", x_train.shape



model = CreateModel()

epoch = 5
for i in range(epoch):
    model.fit(x_train, y_train, batch_size=10, nb_epoch=25, verbose=1)
    save_loader.SaveToObject(model, 'model.mdl')
    score = model.evaluate(x_train, y_train, verbose=0)
    print "train score", score

    y_test = model.predict(x_test)
    print y_test

    result = pandas.DataFrame(test_data.index, columns=['test_id'])
    result['is_duplicate'] = y_test
    result.to_csv('result.csv', index=False)