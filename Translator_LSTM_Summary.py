import numpy as np
import pandas
from gensim.models import word2vec
import pickle_file_manager
from gensim.summarization import summarize
from gensim.summarization import keywords

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

def TextToSCodeNorm(data, modelName, lenght, maxQLen=5, featuresLen = 50):
    res_data = np.zeros((lenght, 2*maxQLen, featuresLen), dtype=np.float16)
    module = pickle_file_manager.LoadFromObject('NModule_' + modelName + '.mdl')
    cmin = pickle_file_manager.LoadFromObject('NMin_' + modelName +  '.mdl')
    word2vec_model = word2vec.Word2Vec.load(modelName + '.mdl')
    print('word2vec_model loaded')


    for j in range(lenght):
        res_row = res_data[j]
        words1 = data.iloc[j, 0]
        words2 = data.iloc[j, 1]

        # keys_1 = keywords(words1, ratio=0.9)
        # keys_2 = keywords(words1, ratio=0.9)
        words1 = words1 + ' ' + words2
        words2 = words2 + ' ' + words2 + ' ' + words2

        print words1
        print words2

        summary_1 = summarize(words1,  split=True)
        summary_2 = summarize(words2,  split=True)


        print summary_1
        print summary_2
        exit()

        for w in summary_1:
            if (w in word2vec_model.wv.vocab):
                temp = (word2vec_model.wv[w] + abs(cmin)) / module
                res_row[time, :] = temp

        for w in summary_2:
            if (w in word2vec_model.wv.vocab):
                temp = (word2vec_model.wv[w] + abs(cmin)) / module
                res_row[time + maxQLen, :] = temp

        if (j % 10000 == 0):
            print j

    # pickle_file_manager.SaveToObject()
    return res_data