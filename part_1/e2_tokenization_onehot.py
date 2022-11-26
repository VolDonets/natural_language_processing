import numpy as np
import pandas as pd


sentence = """Thomas Jefferson bean building Monticello at the age of 26."""
print(sentence.split())

token_sequence = str.split(sentence)
vocab = sorted(set(token_sequence))
print(', '.join(vocab))

num_tokens = len(token_sequence)
vocab_size = len(vocab)
onehot_vectors = np.zeros((num_tokens, vocab_size), int)

for i, word in enumerate(token_sequence):
    onehot_vectors[i, vocab.index(word)] = 1

print(onehot_vectors)

df = pd.DataFrame(onehot_vectors, columns=vocab)
print(df)
print()

df[df == 0] = ''
print(df)
print()

# word counting in a sentence
sentence_bow = {}
for token in sentence.split():
    sentence_bow[token] = 1
print(sentence_bow)

# pd.Series as corpus of texts:
print('\npd.Series as corpus of texts')
df_corpus = pd.DataFrame(pd.Series(dict([(token, 1) for token in sentence.split()])), columns=['sent']).T
print(df_corpus)

sentences = sentence + "\n"
sentences += "Constructions was done mostly by local masons and carpenters.\n"
sentences += "He moved into the South Pavilion in 1770.\n"
sentences += "Turning Monticello into a neoclassical masterpiece was Jefferson's obsession."
corpus = {}
for i, sent in enumerate(sentences.split('\n')):
    corpus['sent{}'.format(i)] = dict((tok, 1) for tok in sent.split())
df_big_corpus = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T
print()
print(df_big_corpus[df_big_corpus.columns[:10]])


