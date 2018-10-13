import numpy as np
import pandas
from gensim.models import word2vec
import pickle_file_manager

def TextToCode(data, modelName, lenght, maxQLen=20, featuresLen = 50):

    res_data = np.zeros((lenght, 2 * maxQLen, featuresLen), dtype=np.float16)
    word2vec_model = word2vec.Word2Vec.load(modelName + '.mdl')
    print('word2vec_model loaded')


    for j in range(lenght):
        res_row = res_data[j]
        words1 = data.iloc[j, 0].split(' ')
        words2 = data.iloc[j, 1].split(' ')

        l1 = len(words1)
        if(l1>maxQLen):
            l1 = maxQLen
        l2 = len(words2)
        if(l2>maxQLen):
            l2 = maxQLen

        # lenght - 1 because last word is empty ''. i don't know why
        already_uses = []
        for time in xrange(l1 - 1):
            w = words1[time]
            if (w in word2vec_model.wv.vocab):
                res_row[time, :] = word2vec_model.wv[w]

        for time in xrange(l2 - 1):
            w = words2[time]
            if (w in word2vec_model.wv.vocab):
                res_row[time + maxQLen, :] = word2vec_model.wv[w]

        if (j % 10000 == 0):
            print j

    return res_data


def TextToCodeNorm(data, modelName, lenght, maxQLen=20, featuresLen = 50):

    res_data = np.zeros((lenght, 2 * maxQLen, featuresLen), dtype=np.float16)
    module = pickle_file_manager.LoadFromObject('NModule_' + modelName + '.mdl')
    cmin = pickle_file_manager.LoadFromObject('NMin_' + modelName +  '.mdl')
    word2vec_model = word2vec.Word2Vec.load(modelName + '.mdl')
    print('word2vec_model loaded')


    for j in range(lenght):
        res_row = res_data[j]
        words1 = data.iloc[j, 0].split(' ')
        words2 = data.iloc[j, 1].split(' ')

        l1 = len(words1)
        if(l1>maxQLen):
            l1 = maxQLen
        l2 = len(words2)
        if(l2>maxQLen):
            l2 = maxQLen

        # print words1
        # print words2

        # lenght - 1 because last word is empty ''. i don't know why
        already_uses = []
        for time in xrange(l1 - 1):
            w = words1[time]
            if (w in word2vec_model.wv.vocab):
                temp = (word2vec_model.wv[w] + abs(cmin)) / module
                res_row[time, :] = temp

        for time in xrange(l2 - 1):
            w = words2[time]
            if (w in word2vec_model.wv.vocab):
                temp = (word2vec_model.wv[w] + abs(cmin)) / module
                res_row[time + maxQLen, :] = temp

        if (j % 10000 == 0):
            print j

    return res_data


def TextToCodeByList(data, modelName, lenght, maxQLen=20, featuresLen = 50):
    module = pickle_file_manager.LoadFromObject('NModule_' + modelName + '.mdl')
    cmin = pickle_file_manager.LoadFromObject('NMin_' + modelName +  '.mdl')
    word2vec_model = word2vec.Word2Vec.load(modelName + '.mdl')
    print('word2vec_model loaded')

    zero_word = np.zeros(50)
    samples = []

    for j in range(lenght):
        words = []
        words1 = data.iloc[j, 0].split(' ')
        words2 = data.iloc[j, 1].split(' ')

        l1 = len(words1)
        if(l1>maxQLen):
            l1 = maxQLen
        l2 = len(words2)
        if(l2>maxQLen):
            l2 = maxQLen

        # print words1
        # print words2

        # lenght - 1 because last word is empty ''. i don't know why
        for time in xrange(l1 - 1):
            w = words1[time]
            if (w in word2vec_model.wv.vocab):
                temp = (word2vec_model.wv[w] + abs(cmin)) / module
                words.append(temp)
            else:
                words.append(zero_word)

        for time in xrange(l2 - 1):
            w = words2[time]
            if (w in word2vec_model.wv.vocab):
                temp = (word2vec_model.wv[w] + abs(cmin)) / module
                words.append(temp)
            else:
                words.append(zero_word)

        samples.append(words)

        if (j % 10000 == 0):
            print j

    # res_data = np.asarray(samples, dtype=np.float16)
    res_data = np.reshape(samples, (lenght, 40, 50))
    print res_data.shape
    exit()

    return res_data