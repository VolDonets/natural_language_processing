sentence = "Claude Monet painted the Grand Canal of Venice in 1806."

# for skip-gram n=2
# if we select word 'painted' as input word -- w_t
# surrounding words: w_t-1 = 'Monet', w_t-2 = 'Claude'
#                    w_t+1 = 'the', w_t+2 = 'Grand'

# and for the model training:
# one-hot vector of words as input
# n-hidden neurons
# softmax output

# one-hot               n hidden neurons          softmax output
#   0                      ()                        0.974
#   1                      ()                        0.001
#   0                      ...                       0.002
#  ...                     ()                         ...
#   0                                                0.001

# And the last one getting word vectors
# 1. skip the last layer (softmax) of the neural network.
# 2. Get weight matrix where columns are weights for each layer
#    and rows -- weights for the next neuron in hidden layer
#    This matrix called 'word vector embedding'
# 3. Do dot product between one-hot vector and weight matrix
#    and get word vector

# + works well with small corpora
# + work well with rare terms.
