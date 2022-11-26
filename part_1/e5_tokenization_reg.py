import re


sentence = """Thomas Jefferson bean building Monticello at the age of 26."""
re_str = r'[-\s.,;!?]+'
tokens = re.split(re_str, sentence)
print(tokens)

sentence2 = """Test boba , a. Big; 3? 2! .J! 24.2, 34,2 283.9 1?"""
tokens = re.split(re_str, sentence2)
print(tokens)

# how regular expressions work
# [] -- indicates a character class
# []+ -- means that match must contain one or more characters
# \s -- shortcut to a predefined character class: all whitespaces ' ', '\t', '\n', '\r', '\f'
# [0-9] -- matches any digit ~ [0123456789]
# [a-z] -- matches all lowercase letters
# [a-zA-Z] -- would mathc any underscore character '_' or letter English alphabet (upper- or lowercase)
# [-] -- means hyphen '-' symbol not a range or [\-]
# () -- used for forcing regular expression to match the entire expression withing the parentheses
#       before try to match the characters that follow the parentheses.

pattern = re.compile(r"([-\s.,:!?])+")
tokens = pattern.split(sentence)
print()
print(tokens)
print([x for x in tokens if x and x not in '- \t\n\r\f.,;!?'])
