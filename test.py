import re
import main
import tqdm


with open("script.txt", "r") as f:
    lines = f.readlines()

script = re.compile(r"[a-zA-Z]\b")

hash_map = main.HashMap()
hash_map.supress_prints = True
for i in tqdm.tqdm(lines):
    hash_map.add("".join(script.findall(i)))