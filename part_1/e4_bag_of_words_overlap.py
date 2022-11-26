import numpy as np
import pandas as pd


sentence = """Thomas Jefferson bean building Monticello at the age of 26."""
sentences = sentence + "\n"
sentences += "Constructions was done mostly by local masons and carpenters.\n"
sentences += "He moved into the South Pavilion in 1770.\n"
sentences += "Turning Monticello into a neoclassical masterpiece was Jefferson's obsession."
corpus = {}
for i, sent in enumerate(sentences.split('\n')):
    corpus['sent{}'.format(i)] = dict((tok, 1) for tok in sent.split())
df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T
print(df[df.columns[:10]])

df = df.T
print('\nBag-of-words overlap:')
print(df.sent0.dot(df.sent1))
print(df.sent0.dot(df.sent2))
print(df.sent0.dot(df.sent3))

print('\nshow the same word for the sent0 and sent3:')
print([(k, v) for (k, v) in (df.sent0 & df.sent3).items() if v])
