import main
import tqdm
from random import choice, randint
from string import ascii_lowercase
import time

strt = time.time()
lines = ["".join([choice(ascii_lowercase) for j in range(randint(1,15))]) for _ in range(1000000)]

print("Generated words, sorting")

a = main.HashMap()

lines.sort(key=lambda x: a.l1Hash(x))

print("Sorting words")

a.words_in(lines)

with open("Times.txt", "a") as f:
    f.write(str(time.time()-strt))
print()