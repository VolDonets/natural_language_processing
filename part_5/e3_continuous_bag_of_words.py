# In CBOW approach you're trying to predict the center word based on
# the surrounding words.
# Here used a multi-hot vector of all surrounding pairs.

# for CBOW n=2
# if we select word 'painted' as output word -- w_t
# surrounding words: w_t-1 = 'Monet', w_t-2 = 'Claude'
#                    w_t+1 = 'the', w_t+2 = 'Grand'
# surrounding words are used as input

# and for the model training:
# multi-hot vector of words as input
# n-hidden neurons
# softmax output

# multi-hot               n hidden neurons          softmax output
#   1                      ()                        0.002
#   1                      ()                        0.001
#   0                      ...                       0.974
#  ...                     ()                         ...
#   0                                                0.001

# + show higher accuracies for frequent words
# + is much faster to train
