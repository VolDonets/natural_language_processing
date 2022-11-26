from nltk.tokenize import RegexpTokenizer

sentence = """Thomas Jefferson bean building Monticello at the age of 26."""
tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
print(tokenizer.tokenize(sentence))


from nltk.tokenize import TreebankWordTokenizer
tokenizer2 = TreebankWordTokenizer()
print(tokenizer2.tokenize(sentence))


# casual tokenizer for dealing this text from social networks such as Twitter and Facebook
from nltk.tokenize.casual import casual_tokenize
message = """RT @TJMonticello Best day everrrrrrr at Monticello. Awesommmmmmeeeeeeee day :*)"""
print(casual_tokenize(message))
print(casual_tokenize(message, reduce_len=True, strip_handles=True))
