# for query: "She invented something to do with physics in Europe
#            in the early 20th century"
# >>> answer_vector = wv['woman'] + wv['Europe'] + wv['physics'] + wv['scientist']
# >>> answer_vector = wv['woman'] + wv['Europe'] + wv['physics'] +\
#                     wv['scientist'] - wv['male'] - 2 * wv['man']

# for query: "Who is to nuclear physics what Louis Paster is to germs?"
# >>> answer_vector = wv['Lois_Pasteur'] - wv['germs'] + wv['physics']
# >>> wv['Marie_Curie'] - wv['science'] + wv['music']

import numpy as np
from nlpia.book.examples.ch06_nessvectors import *

print(nessvector('Marie_Curie').round(2))

