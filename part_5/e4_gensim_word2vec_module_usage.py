# The first way to get gensim word2vec module
from nlpia.data.loaders import get_data
word_vectors = get_data('word2vec')

# The second way to get gensim ..., 'roll your own'
# from gensim.models.keyedvectors import KeyedVectors
# word_vectors = KeyedVectors.load_word2vec_format('path/to/GoogleNews-negative300.bin.gz', binary=True)

# The second way + memory saving
# from gensim.models.keyedvectors import KeyedVectors
# word_vectors = KeyedVectors.load_word2vec_format('path/to/GoogleNews-negative300.bin.gz',
#                               binary=True, limit=200_000)

ms = word_vectors.most_similar(positive=['cooking', 'potatoes'], topn=5)
print("Most similar words to 'cooking' & 'potatoes':")
print(ms)

ms = word_vectors.most_similar(positive=['germany', 'france'], topn=5)
print("\nMost similar words to 'germany' & 'france':")
print(ms)

dm = word_vectors.doesnt_match('potatoes milk cake computer'.split())
print("\nDetermine unrelated terms between ['potatoes', 'milk', 'cake', 'computer']:")
print(dm)

calc = word_vectors.most_similar(positive=['king', 'woman'], negative=['man'], topn=5)
print("\nProcessing calculus for 'king' + 'woman' - 'man' = 'queen':")
print(calc)

sim = word_vectors.similarity('princess', 'queen')
print("\nCalculating similarity: 'princess' & 'queen'")
print(sim)

wv = word_vectors['phone']
# wv = word_vectors.get('phone')
print('\nGetting a raw word vector for word "phone":')
print(wv)
