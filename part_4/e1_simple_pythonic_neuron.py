import numpy as np
from random import random


example_input = [1., .2, .1, .05, .2]
example_weights = [.2, .12, .4, .6, .9]

input_vector = np.array(example_input)
weights = np.array(example_weights)
bias_weight = .2
activation_level = np.dot(input_vector, weights) + (bias_weight * 1.)
print(activation_level)

threshold = 0.5
perceptron_output = 1. if activation_level >= threshold else 0.
print(perceptron_output)

expected_output = 0
new_weights = []
for i, x in enumerate(example_input):
    new_weights.append(weights[i] + (expected_output - perceptron_output) * x)
weights = np.array(new_weights)
print("Example weights:", example_weights)
print('Weights:', weights)

sample_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected_results = [0, 1, 1, 1]
activation_threshold = 0.5

np.random.random(1_000_000)
weights = np.random.random(2) / 1000.    # gen small random numbers
print('\n\n\nWeights: ', weights)
bias_weight = np.random.random() / 1000.
print('Bias weight:', bias_weight, '\n')

for idx, sample in enumerate(sample_data):
    input_vector = np.array(sample)
    activation_level = np.dot(input_vector, weights) + (bias_weight * 1.)
    perceptron_output = 1 if activation_level > activation_threshold else 0
    print('Pr: {}, Exp: {}'.format(perceptron_output, expected_results[idx]))

for iteration_num in range(5):
    correct_answers = 0
    for idx, sample in enumerate(sample_data):
        input_vector = np.array(sample)
        weights = np.array(weights)
        activation_level = np.dot(input_vector, weights) + (bias_weight * 1.)
        perceptron_output = 1. if activation_level > activation_threshold else 0.
        if perceptron_output == expected_results[idx]:
            correct_answers += 1
        new_weights = []
        for i, x in enumerate(sample):
            new_weights.append(weights[i] + (expected_results[idx] - perceptron_output) * x)
        bias_weight = bias_weight + ((expected_results[idx] - perceptron_output) * 1.)
    print('{} correct answers of 4, for iteration {}'.format(correct_answers, iteration_num))


