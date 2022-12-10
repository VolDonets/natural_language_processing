import os
import pandas as pd
import numpy as np
import seaborn
from nlpia.loaders import get_data
from gensim.models.word2vec import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


wv = get_data('word2vec')
print('len wv.vocab:', len(wv))

# ! It doesn't work in gensim 4.+
# vocab = pd.Series(wv.vocab)
# print(vocab.iloc[1_000_000:100_006])

print("count weights in word-vector:", len(wv['Illini']), '\n')

euclid_dist = np.linalg.norm(wv['Illinois'] - wv['Illini'])
cos_similarity = np.dot(wv['Illinois'], wv['Illini'] /
                        (np.linalg.norm(wv['Illinois'] * np.linalg.norm(wv['Illini']))))
print("Euclidean distance between 'Illinois' & 'Illini':", euclid_dist)
print("Cosine SIMILARITY between 'Illinois' & 'Illini':", cos_similarity)
print("Cosine DISTANCE between 'Illinois' & 'Illini':", 1. - cos_similarity)

cities = get_data('cities')
print("\n\nNow let's get cities data:")
print(cities.head(1).T)

# let's consider only on US cities:
us = cities[(cities.country_code == 'US') & (cities.admin1_code.notnull())].copy()
states = pd.read_csv('http://www.fonz.net/blog/wp-content/uploads/2008/04/states.csv')
states = dict(zip(states.Abbreviation, states.State))

us['city'] = us.name.copy()
us['st'] = us.admin1_code.copy()
us['state'] = us.st.map(states)
print('\n\nUSA cities:')
print(us[us.columns[-3:]].head())

vocab = np.concatenate([us.city, us.st, us.state])
vocab = np.array([word for word in vocab if word in wv])
print('\n\nSelected vocabulary:')
print(vocab[:5])

city_plus_state = []
for c, state, st in zip(us.city, us.state, us.st):
    if c not in vocab:
        continue
    row = []
    if state in vocab:
        row.extend(wv[c] + wv[state])
    else:
        row.extend(wv[c] + wv[st])
    city_plus_state.append(row)

us_300D = pd.DataFrame(city_plus_state)

pca = PCA(n_components=2)
us_300D = get_data('cities_us_wordvectors')
us_2D = pca.fit_transform(us_300D.iloc[:, :300])

# visualize cities
plt.scatter(us_2D[:, 0], us_2D[:, 1])
plt.title('All cities')
plt.show()

us_2D = us_2D[:200]
plt.scatter(us_2D[:, 0], us_2D[:, 1])
plt.title('200 cities')
plt.show()

# !!! ploty doesn't work and I'm too lazy to fix it.
# df = get_data('cities_us_wordvectors_pca2_meta')
# from nlpia.plots import offline_plotly_scatter_bubble
# html = offline_plotly_scatter_bubble(
#     df.sort_values('population', ascending=False)[:350].copy()\
#         .sort_values('population'),
#     filename='..src/plotly_scatter_bubble.html',
#     x='x', y='y',
#     size_col='population', text_col='name', category_col='timezone',
#     xscale=None, yscale=None, # 'log' or None
#     layout={}, marker={'sizeref': 3000})

