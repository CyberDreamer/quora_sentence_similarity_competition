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

    pickle_file_manager.SaveToObject(module,'NModule_W2V_Model_Quora_50.mdl')
    pickle_file_manager.SaveToObject(cmin, 'NMin_W2V_Model_Quora_50.mdl')

    print module

    # for key, value in model.wv.vocab.items():
    #     temp = (model.wv[key] + abs(cmin)) / module
    #     print key, ' code:', temp

    print 'Normalization finished...'



# ----------------------- TEST --------------------------
# sLength = 404290
# sLength = 10000
# source = pandas.read_csv('train.csv').head(n=sLength)
# source = textColumnsToLowcase(source)
# source = specSymbolReplacer(source)
# source = source[['question1', 'question2']].dropna()
# q1_list = source['question1'].str.split().values.tolist()
# q2_list = source['question2'].str.split().values.tolist()
#
# sentences = q1_list + q2_list
# print 'Data prepeared...'


# pickle_file_manager.SaveToObject(sentences, 'Quora_sentences.dat')
# print 'Data saved...'

sentences = pickle_file_manager.LoadFromObject('Quora_sentences.dat')
print 'Data load...'
# Get_Statisctic(sentences)

model = word2vec.Word2Vec(sentences, size=50, window=14, min_count=5, workers=4)
Normalization(model)


# model = word2vec.Word2Vec.load('word2vec_quora_set_model.mdl')
# print 'word2vec_model loaded'

model.save('W2V_Model_Quora_50.mdl')
print '<computer> code: ', model.wv['why']
print 'what and why similar: ', model.similarity('why', 'what')
print 'what and girl similar: ', model.similarity('girl', 'what')
