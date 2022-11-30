import numpy as np

topic = {}
tfidf = dict(list(zip('cat dog apple lion NYC love'.split(), np.random.rand(6))))

topic['petness'] = .3 * tfidf['cat'] + \
                   .3 * tfidf['dog'] + \
                   .0 * tfidf['apple'] + \
                   .0 * tfidf['lion'] - \
                   .2 * tfidf['NYC'] + \
                   .2 * tfidf['love']
topic['animalness'] = .1 * tfidf['cat'] + \
                   .1 * tfidf['dog'] - \
                   .5 * tfidf['apple'] + \
                   .1 * tfidf['lion'] + \
                   .1 * tfidf['NYC'] - \
                   .1 * tfidf['love']
topic['cityness'] = .0 * tfidf['cat'] - \
                   .1 * tfidf['dog'] + \
                   .2 * tfidf['apple'] - \
                   .1 * tfidf['lion'] + \
                   .5 * tfidf['NYC'] + \
                   .1 * tfidf['love']

print(tfidf)
print(topic)

word_vector = {}
word_vector['cat'] = .3 * topic['petness'] + \
                     .1 * topic['animalness'] + \
                     .0 * topic['cityness']
word_vector['dog'] = .3 * topic['petness'] + \
                     .1 * topic['animalness'] + \
                     .1 * topic['cityness']
word_vector['apple'] = .0 * topic['petness'] - \
                       .1 * topic['animalness'] + \
                       .2 * topic['cityness']
word_vector['lion'] = .0 * topic['petness'] + \
                      .5 * topic['animalness'] - \
                      .1 * topic['cityness']
word_vector['NYC'] = -.2 * topic['petness'] + \
                      .1 * topic['animalness'] + \
                      .5 * topic['cityness']
word_vector['love'] = .2 * topic['petness'] - \
                      .1 * topic['animalness'] + \
                      .1 * topic['cityness']

print(word_vector)

