import numpy as np
import pandas
from gensim.models import word2vec
import pickle_file_manager


def textColumnsToLowcase(data):
    data[['question1', 'question2']] = data[['question1', 'question2']]\
        .apply(lambda column: column.str.lower())

    return data

def specSymbolReplacer(data):
    # data = data[['question1', 'question2']].replace('[^a-zA-Z0-9]', ' ', regex=True)
    data = data[['question1', 'question2']].replace('[^a-zA-Z]', ' ', regex=True)
    # data = data[['question1', 'question2']].replace('[]\'', '')
    return data


def Get_Statisctic(sentences):
    lenght = len(sentences)
    lenght = lenght/4
    print lenght
    sizes = np.zeros(lenght)
    for ii in range(lenght):
        sizes[ii] = len(sentences[ii])

    mean = np.mean(sizes)
    min = np.min(sizes)
    max = np.max(sizes)

    print 'MEAN words count: ', mean
    print 'MIN words count: ', min
    print 'MAX words count: ', max

def Normalization(model):
    print len(model.wv.vocab)
    total_vectors = []
    for key, value in model.wv.vocab.items():
        # print key, ' code:', model.wv[key]
        total_vectors.append(model.wv[key])

    cmax = np.array(total_vectors).max(axis=0)
    # print cmax
    cmin = np.array(total_vectors).min(axis=0)
    # print cmin
    # print ''
    # print ''

    module = abs(cmin) + cmax

    pickle_file_manager.SaveToObject(module,'NModule_' + modelName + '.mdl')
    pickle_file_manager.SaveToObject(cmin, 'NMin_' + modelName +  '.mdl')

    print module

    # for key, value in model.wv.vocab.items():
    #     temp = (model.wv[key] + abs(cmin)) / module
    #     print key, ' code:', temp

    print 'Normalization finished...'


def TextToCode(data, modelName, lenght, groupLen=6, codeLen=50, maxQLen=60):

    res_data = np.zeros((lenght, 2, 2 * groupLen, codeLen), dtype=np.float16)

    word2vec_model = word2vec.Word2Vec.load(modelName + '.mdl')
    print('word2vec_model loaded')
    module = pickle_file_manager.LoadFromObject('NModule_' + modelName + '.mdl')
    cmin = pickle_file_manager.LoadFromObject('NMin_' + modelName +  '.mdl')

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
                # print 'merge: ', w1, " + ", w2
                local_similarity = 0.
                if (w1 in word2vec_model.wv.vocab and w2 in word2vec_model.wv.vocab):
                    local_similarity = word2vec_model.wv.similarity(w1, w2)

                if (hight_similarity_count < groupLen and local_similarity > 0.8):
                    if (w1 not in already_uses and w2 not in already_uses):
                        s1 = (word2vec_model.wv[w1] + abs(cmin)) / module
                        s2 = (word2vec_model.wv[w2] + abs(cmin)) / module
                        res_row[0, hight_similarity_count, :] = s1
                        res_row[1, hight_similarity_count, :] = s2

                        already_uses.append(w1)
                        already_uses.append(w2)

                        # print 'H  ', w1
                        # print 'H  ', w2
                        # print res_row[0:100, hight_similarity_count]
                        # print res_row[100:200, hight_similarity_count]

                        hight_similarity_count += 1
                        break

                if (low_similarity_count < groupLen and local_similarity > 0 and local_similarity < 0.2):
                    if (w1 not in already_uses and w2 not in already_uses):
                        s1 = (word2vec_model.wv[w1] + abs(cmin)) / module
                        s2 = (word2vec_model.wv[w2] + abs(cmin)) / module

                        res_row[0, groupLen + low_similarity_count, :] = s1
                        res_row[1, groupLen + low_similarity_count, :] = s2

                        already_uses.append(w1)
                        already_uses.append(w2)

                        # print 'L  ', w1
                        # print 'L  ', w2
                        low_similarity_count += 1

            if (low_similarity_count >= groupLen and hight_similarity_count >= groupLen):
                break

        if (j % 10000 == 0):
            print j

    return res_data


def CreateSentenceFromSourceData(filenameSource, filenameOut):
    sLength = 800000
    # sLength = 404290
    # sLength = 10000
    source = pandas.read_csv(filenameSource)[0:500000]
    source = textColumnsToLowcase(source)
    source = specSymbolReplacer(source)
    source = source[['question1', 'question2']].dropna()
    q1_list = source['question1'].str.split().values.tolist()
    q2_list = source['question2'].str.split().values.tolist()

    sentences = q1_list + q2_list
    print 'Data prepeared...'

    # pickle_file_manager.SaveToObject(sentences, filenameOut)
    print 'Data saved...'

    return sentences

if __name__ == '__main__':
    # ----------------------- TEST --------------------------
    # sentences_1 = CreateSentenceFromSourceData('train.csv', 'Quora_sentences_train.dat')
    sentences_2 = CreateSentenceFromSourceData('test.csv', 'Quora_sentences_test_2.dat')
    # exit()
    print 'Wooooop'
    # sentences_1 = pickle_file_manager.LoadFromObject('Quora_sentences_train.dat')
    # sentences_2 = pickle_file_manager.LoadFromObject('Quora_sentences_test_2.dat')
    # sentences_3 = word2vec.Text8Corpus('/tmp/text8')


    # print len(sentences_1)
    print len(sentences_2)

    # full_sentence = sentences_1 + sentences_2
    print 'Data load...'
    # Get_Statisctic(sentences)

    modelName = 'W2V_Model_QuoraTTT_50'
    model = word2vec.Word2Vec(full_sentence, size=50, window=5, min_count=3, workers=4)
    model = word2vec.Word2Vec.load('W2V_Model_QuoraTT_50.mdl')
    model.train(sentences_2)
    Normalization(model)
    # print 'word2vec_model loaded'

    model.save(modelName + '.mdl')
    print '<computer> code: ', model.wv['why']
    print 'what and why similar: ', model.similarity('why', 'what')
    print 'what and girl similar: ', model.similarity('girl', 'what')
