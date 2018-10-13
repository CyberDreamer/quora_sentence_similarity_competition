import numpy as np
import pandas
from gensim.models import word2vec


def textColumnsToLowcase(data):
    data[['question1', 'question2']] = data[['question1', 'question2']]\
        .apply(lambda column: column.str.lower())

    return data

def specSymbolReplacer(data):
    data[['question1', 'question2']] = data[['question1', 'question2']].replace('[^a-zA-Z0-9]', ' ', regex=True)
    return data


def TextToCode(row):
    q1 = row['question1']
    q2 = row['question2']

    words1 = q1.split(' ')
    words2 = q2.split(' ')

    l1 = len(words1)
    l2 = len(words2)

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



# ----------------------- TEST --------------------------
# lenght = 2345796
sLength = 100000
# source = pandas.read_csv('test.csv').head(n=sLength)
source = pandas.read_csv('train.csv').head(n=sLength)
model = word2vec.Word2Vec.load('word2vec_quora_set_model.mdl')
# print '<computer> code: ', model.wv['computer']
print 'word2vec_model loaded'


# step = 586449
# step = 100000
# ss = 4
# for m in range(ss):
#     start = step*m
#     end = start + step
# test_data = source_test_data[['question1', 'question2', 'similarity']].iloc[start:end]
y_train = source[['is_duplicate']].values
source = source[['question1', 'question2']]
source = textColumnsToLowcase(source)
source = specSymbolReplacer(source)
source = source[['question1', 'question2']].fillna("Empty")

print 'Data prepeared...'

source['similarity'] = pandas.Series(index=source.index)
test_data = source.apply(TextToCode, axis=1)

    # print m+1, "/", ss, "  part predicted..."

y_test = test_data[['similarity']].values

accuracy = 0.

for ii in range(sLength):
    # print 'train: ', y_train[ii]
    # print 'test: ', y_test[ii]
    # print ''
    if(abs(y_train[ii]-y_test[ii])<0.5):
        accuracy+=1

print accuracy/sLength
exit()
# print test_data

print y_test
print "max:    ", max(y_test)
print "min:    ", min(y_test)
print "median: ", np.median(y_test)

result = pandas.DataFrame(test_data.index, columns=['test_id'])
result['is_duplicate'] = y_test

resultName = 'result_' + '_W2V_Greedy' + '.csv'
result.to_csv(resultName, index=False)
print "Result saved..."