import numpy as np
import pandas
from gensim.models import word2vec
import pickle_file_manager


def TextToCode(data, modelName, lenght, maxQLen=30):

    res_data = np.zeros((lenght, 1, maxQLen, maxQLen), dtype=np.float16)

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

        hight_similarity_count = 0
        low_similarity_count = 0

        # lenght - 1 because last word is empty ''. i don't know why
        already_uses = []
        for index_w1 in xrange(l1 - 1):
            w1 = words1[index_w1]
            for index_w2 in xrange(l2 - 1):
                w2 = words2[index_w2]

                local_similarity = 0.
                if (w1 in word2vec_model.wv.vocab and w2 in word2vec_model.wv.vocab):
                    local_similarity = word2vec_model.wv.similarity(w1, w2)

                    res_row[0,index_w1, index_w2] = local_similarity

        if (j % 10000 == 0):
            print j

    return res_data