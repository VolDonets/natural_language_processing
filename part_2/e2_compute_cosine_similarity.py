import math


def cosine_sim(vec1: dict, vec2: dict):
    """Convert our dicts into lista for easier matching."""
    vec1 = list(vec1.values())
    vec2 = list(vec2.values())

    dot_product = 0.0
    for i, v in enumerate(vec1):
        dot_product += v * vec2[i]

    mag_1 = math.sqrt(sum([x ** 2 for x in vec1]))
    mag_2 = math.sqrt(sum([x ** 2 for x in vec2]))

    return dot_product / (mag_1 * mag_2)

# cosine similarity values:
# 1 -- means documents have the same words and with high probability
#      have the same topic (vectors point in the same direction)
# 0 -- means documents don't have same words and with high probability
#      don't have the same topic (vectors are orthogonal, perpendicular)
# -1 -- means documents are anti-similar (completely opposite), normally
#       it doesn't happen cause tf (counts) can't be negative, but it's
#       possible for 'concept of words/topics that are "opposite"
