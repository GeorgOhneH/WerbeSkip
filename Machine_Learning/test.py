import re

m = re.match(r"([a-zA-Z]+)([0-9]+)","A11")
print(int(m.group(2)))
print(ord(m.group(1)) + int(m.group(2)) / 10 ** len(m.group(2)))