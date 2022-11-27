from nltk.util import ngrams
import re


sentence = """Thomas Jefferson bean building Monticello at the age of 26."""
pattern = re.compile(r"([-\s.,;!?])+")
tokens = pattern.split(sentence)
tokens = [x for x in tokens if x and x not in '- \t\n.,;!?']
print(tokens, '\n')

two_grams = list(ngrams(tokens, 2))
print(two_grams, '\n')
print(list(ngrams(tokens, 3)), '\n')

print([" ".join(x) for x in two_grams])
