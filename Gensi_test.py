from gensim.summarization import summarize
from gensim.models import word2vec
from gensim.summarization import keywords

# ----------------------- WIKI --------------------------

import wikipedia
# result = wikipedia.summary("GitHub")
# print result


s = "What is the best way to parse Wikipedia articles using Python?"
keys = keywords(s, ratio=0.8, split=True)
print keys

for k in keys:
    result = wikipedia.summary(k)
    print result


exit()

# ----------------------- SIMILARITY ---------------------

# text = "Thomas A. Anderson is a man living two lives. By day he is an " + \
#     "average computer programmer and by night a hacker known as " + \
#     "Neo. Neo has always questioned his reality, but the truth is " + \
#     "far beyond his imagination. Neo finds himself targeted by the " + \
#     "police when he is contacted by Morpheus, a legendary computer " + \
#     "hacker branded a terrorist by the government. Morpheus awakens " + \
#     "Neo to the real world, a ravaged wasteland where most of " + \
#     "humanity have been captured by a race of machines that live " + \
#     "off of the humans' body heat and electrochemical energy and " + \
#     "who imprison their minds within an artificial reality known as " + \
#     "the Matrix. As a rebel against the machines, Neo must return to " + \
#     "the Matrix and confront the agents: super-powerful computer " + \
#     "programs devoted to snuffing out Neo and the entire human " + \
#     "rebellion. "
#
# print 'Input text:'
# print text
#
# print 'Summary:'
# print summarize(text)
#
# exit()


# ----------------------- Word2Vec ---------------------
# sentences = word2vec.Text8Corpus('text8')
# model = word2vec.Word2Vec(sentences, size=200)
# model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
# model.save('word2vec_model.mdl')
model = word2vec.Word2Vec.load('word2vec_model.mdl')

# print '<computer> code: ', model.wv['computer']
print model.wv.similarity('computer', 'machine')

# print model.most_similar(['man'])