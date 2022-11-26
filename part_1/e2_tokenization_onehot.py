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
