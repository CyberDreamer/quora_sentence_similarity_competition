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



def textColumnsToLowcase(data):
    data[['question1', 'question2']] = data[['question1', 'question2']]\
        .apply(lambda column: column.str.lower())

    return data

def specSymbolReplacer(data):
    data[['question1', 'question2']] = data[['question1', 'question2']].replace('[^a-zA-Z0-9]', ' ', regex=True)
    return data

# ['test_id', 'question1', 'question2']
# .head(n=5000)
train_data = pandas.read_csv('train.csv')
test_data = pandas.read_csv('test.csv')


train_data = textColumnsToLowcase(train_data)
test_data = textColumnsToLowcase(test_data)

train_data = specSymbolReplacer(train_data)
test_data = specSymbolReplacer(test_data)

vectorizer = TfidfVectorizer()
train_data = train_data[['question1', 'question2', 'is_duplicate']].dropna()
y_train = train_data['is_duplicate'].values

test_data = test_data[['question1', 'question2']].fillna("Empty")


train_q1_m = vectorizer.fit_transform(train_data['question1'])
train_q2_m = vectorizer.transform(train_data['question2'])

test_q1_m = vectorizer.transform(test_data['question1'])
test_q2_m = vectorizer.transform(test_data['question2'])




x_train = train_q1_m + train_q2_m
x_test = test_q1_m + test_q2_m

# x_train = x_train.toarray()
# x_test = x_test.toarray()
# print x_train
# print "max features: ", len(vectorizer.vocabulary_)
# print "sparse matrix shape: ", x_train.shape

# model = Pipeline([('poly', PolynomialFeatures(degree=2)), ('Ridge', LinearRegression())])
# model = LogisticRegression()
epoch = 50
neurons = (3,)
model = MLPClassifier(hidden_layer_sizes=neurons, activation="relu", verbose=1, max_iter=epoch, batch_size=128, alpha=0.1)

model.fit(x_train, y_train)
modelName = 'model_E' + str(epoch) + '_NL' + str(neurons) + '.mdl'
save_loader.SaveToObject(model, modelName)
print "Model saved..."
score = model.score(x_train, y_train)
print "train score", score

y_test = model.predict_proba(x_test)[:,1]
print y_test

result = pandas.DataFrame(test_data.index, columns=['test_id'])
result['is_duplicate'] = y_test
resultName = 'result_E' + str(epoch) + '_NL' + str(neurons) + '.csv'
result.to_csv(resultName, index=False)
print "Result saved..."