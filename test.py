import re
import main
import tqdm
from random import choice, randint
from string import ascii_lowercase
import time

strt = time.time()
lines = ["".join([choice(ascii_lowercase) for j in range(randint(1,15))]) for _ in range(10000)]

a = main.HashMap()
a.supress_prints = True

lines.sort(key=lambda x: a.l1Hash(x))

for i in tqdm.tqdm(lines):
    a.add(i)

with open("Times.txt", "a") as f:
    f.write(str(time.time()-strt))
print()