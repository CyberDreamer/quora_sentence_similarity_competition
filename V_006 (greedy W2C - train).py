import numpy as np
import pandas
from gensim.models import word2vec
import Saver_Loader as save_loader


def textColumnsToLowcase(data):
    data[['question1', 'question2']] = data[['question1', 'question2']]\
        .apply(lambda column: column.str.lower())

    return data

def specSymbolReplacer(data):
    # data = data[['question1', 'question2']].replace('[^a-zA-Z0-9]', ' ', regex=True)
    data = data[['question1', 'question2']].replace('[^a-zA-Z]', ' ', regex=True)
    # data = data[['question1', 'question2']].replace('[]\'', '')
    return data


def TextToCode(row):
    q1 = row['question1']
    q2 = row['question2']

    summ_sim = 0
    count = 0

    words1 = q1.split(' ')
    words2 = q2.split(' ')

    l1 = len(words1)
    l2 = len(words2)

    # max = 200
    #
    # if (l1 > max):
    #     l1 = max
    # if (l2 > max):
    #     l2 = max

    ls = np.zeros(l1*l2)

    for index_w1 in range(l1):
        w1 = words1[index_w1]
        for index_w2 in range(l2):
            w2 = words2[index_w2]
            # print 'merge: ', w1, " + ", w2
            local_similarity = 0.5
            if (w1 in model.wv.vocab and w2 in model.wv.vocab):
                local_similarity = model.wv.similarity(w1, w2)

            ls[index_w2 + l2 * index_w1] = local_similarity

    sim = np.average(ls)
    # sim = np.median(ls)
    row['similarity'] = sim
    return row



def AddOneSentence(row, sentences):
    row = str(row)
    # row = row.replace('[]\'','')
    # row = row.replace(']','')
    # row = row.replace('\'','')
    row = row.split()

    # print row
    # exit()
    sentences.append(row)


def Statisctic(sentences, lenght):
    for ii in range(lenght):
        sentences

    mean = np.mean()
    min = np.min()
    max = np.max()

    print 'MEAN words count: ', mean
    print 'MIN words count: ', min
    print 'MAX words count: ', max


# ----------------------- TEST --------------------------
sLength = 404290
# sLength = 1000
source = pandas.read_csv('train.csv').head(n=sLength)
source = textColumnsToLowcase(source)
source = specSymbolReplacer(source)
source = source[['question1', 'question2']].dropna()
# y_train = source['is_duplicate'].values

sentences = []
import string
print len(source)
for ii in range(len(source)):
    row = source[['question1', 'question2']].iloc[ii].values

    q1_row = row[0]
    q2_row = row[1]

    AddOneSentence(q1_row, sentences)
    AddOneSentence(q2_row, sentences)

    if ii % 1000 == 0:
        print ii, '/', len(source)

save_loader.SaveToObject(sentences, 'Quora_sentences.dat')
print 'Data prepeared...'
model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=4, workers=4)
# model = word2vec.Word2Vec.load('word2vec_quora_set_model.mdl')

print 'word2vec_model loaded'

model.save('word2vec_quora_set_model.mdl')
print '<computer> code: ', model.wv['why']
print 'what and why similar: ', model.similarity('why', 'what')
print 'what and girl similar: ', model.similarity('girl', 'what')
