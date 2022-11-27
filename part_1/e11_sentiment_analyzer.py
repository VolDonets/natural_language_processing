from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sa = SentimentIntensityAnalyzer()
# print(sa.lexicon)
# print([(tok, score) for tok, score in sa.lexicon.items()])

sa_res = sa.polarity_scores(text="Python is very readable and it's great for NLP.")
print(sa_res)

sa_res = sa.polarity_scores(text="Python is not a bad choice for most applications.")
print(sa_res)

sa_res = sa.polarity_scores(text="Your messages are horrible for most normal people. Please newer write me again!")
print(sa_res)

print("\n\nSome other testing for VADER SA")
corpus = ["Absolutely perfect! Love it! :-) :-) :-)",
          "Horrible! Completely useless. :-(",
          "It was OK. Some good and some bad things."]
for doc in corpus:
    scores = sa.polarity_scores(doc)
    print(">>>", doc, "<<<\n", scores)
