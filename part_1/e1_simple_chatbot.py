import re

print('--> I <-----------------------------------')
r = "(hi|hello|hey)[ ]*([a-z]*)"

matching_res = re.match(r, 'Hello Rosa', flags=re.IGNORECASE)
print(matching_res)

matching_res = re.match(r, "hi ho, hi ho, it's off to work ...", flags=re.IGNORECASE)
print(matching_res)

matching_res = re.match(r, "hey, what's up", flags=re.IGNORECASE)
print(matching_res)

matching_res = re.match(r, "hEllO, Boba", flags=re.IGNORECASE)
print(matching_res)

print('\n--> II <-----------------------------------')
r = r"[^a-z]*([y]o|[h']?ello|ok|hey|(good[ ])?(morn[gin']{0,3}|"\
    r"afternoon|even[gin']{0,3}))[\s,;:]{1,3}([a-z]{1,20})"
re_greeting = re.compile(r, flags=re.IGNORECASE)

matching_res = re_greeting.match('Hello Rosa')
print(matching_res)

matching_res = re_greeting.match('Hello Rosa').groups()
print(matching_res)

matching_res = re_greeting.match('Good morning Rosa')
print(matching_res)

matching_res = re_greeting.match('Good Manning Rosa')
print(matching_res)

matching_res = re_greeting.match('Good evening Rosa Parks').groups()
print(matching_res)

matching_res = re_greeting.match("Good Morn'n Rosa")
print(matching_res)

matching_res = re_greeting.match('yo Rosa')
print(matching_res)

print('\n--> III <-----------------------------------')
print('Here print greeting')
my_names = set(['rosa', 'rose', 'chatty', 'chatbot', 'bot', 'chatterbot'])
curt_names = set(['hal', 'you', 'u'])
greeter_name = ''

for _ in range(10):
    match = re_greeting.match(input())
    if match:
        at_name = match.groups()[-1]
        if at_name in curt_names:
            print('Good one')
        elif at_name.lower() in my_names:
            print("Hi {}, How are you?".format(greeter_name))
        else:
            print("Wrong greeting")
