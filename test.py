import students_code
from random import choice, randint
from string import ascii_lowercase
import time

strt = time.perf_counter()
lines = ["".join([choice(ascii_lowercase) for j in range(randint(1,15))]) for _ in range(10000)]

print("Generated words, sorting")

a = students_code.HashMap()

print("Sorting words")

a.words_in(lines)

with open("Times.txt", "a") as f:
    f.write(str(time.perf_counter()-strt))
    f.write("\n")
    f.write("\n".join(",".join([i, str(a.lookup_word_count(i))]) for i in sorted(lines, key=lambda x: a.l1Hash(x))))
print()

