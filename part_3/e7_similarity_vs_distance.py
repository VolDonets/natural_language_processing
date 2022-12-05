# For typical similarities and distances:
# similarity = 1. / (1. + distance)
# distance = (1. / similarity) - 1.

# For distances & similarities in range bt 0 & 1 (like probabilities):
# similarity = 1. - distance
# distance = 1. - similarity

# For cosine distance and similarity which are reciprocal of each other:
# import math
# angular_distance = math.acos(cosine_similarity) / math.pi
# distance = 1. / similarity - 1.
# similarity = 1. - distance

