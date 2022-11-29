import nltk

# The Brown corpus is about 3Mb
nltk.download('brown')
from nltk.corpus import brown

print(brown.words()[:10])
print(brown.tagged_words()[:5])
print(len(brown.words()))

from collections import Counter
puncs = set((',', '.', '--', '-', '!', '?', ':', ';', '``', "''", '(', ')', '[', ']'))
word_list = (x.lower() for x in brown.words() if x not in puncs)
token_counts = Counter(word_list)
print('Most common words:')
print(token_counts.most_common(20))

