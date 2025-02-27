import numpy as np
import threading
import warnings
import string
from random import randint, choice
from time import time_ns
from re import sub
from math import log


class WordPair:
    def __init__(self, word, key, number=1):
        self.word = word
        self.number = number
        self.key = key


class HashMap:
    def __init__(self):
        warnings.simplefilter('ignore') # These are obnoxious, and an expected, so we'll just get rid of them
        # This array holds a list of indices
        self.index_array = []
        self.funct = None
        self.array = []
        self.size = 0
        self.supress_prints = False

        self.polyThread = None
        self.threadActive = False
        # Not really meant to be changed, for debugging
        self.l1Hash = lambda x: int("".join([str(ord(i.upper())) for i in list(x)[:min(9, len(x))]]))
    def add(self, item):
        """
        IDEAS TO SPEED UP
        Parrallelize the first for loop in the regression function
        When passing in a lot of data, refrain from actually updating regression until it's done DONE
        """
        
        if len(item) == 0:
            return

        try:
            a = self.__getitem__(item)[0]
            a.number += 1
            return
        except (IndexError, TypeError):
            # adds an item onto the map, then starts the regression calculation seperately
            self.soft_insert(item)
            # Let's run this on a seperate thread, this could get time consuming, and any time saved is something
            self.update()
            
    def update(self):
        print(f"Updating formulas for {len(self.array)} items")
        self.array.sort(key=lambda x: self.l1Hash(x.word))
        self.assert_safe()
        self.polyThread = threading.Thread(target=self.calculate_regression)
        self.polyThread.start()
        self.threadActive = True

    def calculate_regression(self):
        """ This function takes a list and then uses numpy to find a polynomial function that closely models the list
        The X axis is the values passed in, the Y axis are the indices from 0, len(lst)
        """

        lst = [log(i.key, 10) for i in self.array]

        def lagrange_coefficents(x_vals, y_vals):
            n = len(x_vals)
            coeffs = np.poly1d([0])
            for i in range(n):
                p = np.poly1d([1])
                for j in range(n):
                    if i != j:
                        p *= np.poly1d([1, -x_vals[j]]) / (x_vals[i] - x_vals[j])
                coeffs += p * y_vals[i]
            return coeffs

        coes = lagrange_coefficents(lst, list(range(len(lst))))
        self.funct = np.poly1d(coes)

    def soft_insert(self, item, count=1):
        item = sub("[^A-Za-z]", "", item)
        if len(item) == 0:
            return
        
        itemIndex = self.l1Hash(item)
        self.array.append(WordPair(item, itemIndex, count))
        self.array.sort(key=lambda x: x.key)
     
    def words_in(self, words):
        unique_words = sorted(list(set(words)))
        
        for item in unique_words:
            self.soft_insert(item, 0)

        print("Added each unique word, populating counts, this may take a while")
            
        self.update()
        
    
        for item in words:
            self.add(item)

        return len(self.array), 0
                        
    def assert_safe(self):
        """ This method checks to make sure that the regression function finished"""
        if self.threadActive:
            if not self.supress_prints:
                print("Waiting for regression thread to finish")
            time = time_ns()
            self.polyThread.join()
            self.threadActive = False
            if not self.supress_prints:
                print(f"Thread finished in {(time_ns() - time)/1000000000} seconds")

    def __getitem__(self, index):
        index = sub("[^A-Za-z]", "", index)
        if len(index) == 0:
            raise IndexError("Only letter characters are allowed")
        scaled_index = log(self.l1Hash(index),10)
        self.assert_safe()
        if len(self.array) == 0:
            raise IndexError("The Hash Map is empty")
        else:
            # We take the base index, that will likely never have collisions, then run it through our approximated function
            # We calculated in the calculate_regression function
            # We need to check to make sure the item we got is the same as the one we request
            
            result = self.funct(scaled_index)
            
            if result is None:
                raise IndexError("The item is not in the array")

            item = self.array[round(result)]
            if item.key == self.l1Hash(index):
                return item, 1
            else:
                raise IndexError("The item is not in the Hash Map, or there was a mismatch")
    
    def lookup_word_count(self, word):
        item = self.__getitem__(word)
        return item[0].number, item[1]


def words_in(words):
    test_hash = HashMap()
    words.sort()
    return test_hash.words_in(words), test_hash

def lookup_word_count(word, hash):
    item = hash[word]
    return item[0].number, item[1]

if __name__ == "__main__":
    test = HashMap()
    def generate_random_word():
        return ''.join([choice(string.ascii_lowercase) for _ in range(randint(1, 8))])
        
    
    #inwords = ["a", "a", "as", "at"]
    #inwords = [generate_random_word() for _ in range(10)]
    inwords = ['g', 'n', 'n', 'hp', 'ij', 'md', 'ra', 'so', 'ty', 'xu', 'drd', 'fhj', 'gyg', 'hih', 'mfk', 'pae', 'umc', 'xfk', 'zee', 'cxou', 'iwld', 'pdiw', 'zovk', 'clyeb', 'efjsw', 'gxvwc', 'wjoaa', 'yxxut', 'zxrnn', 'etmlxo', 'fthzoy', 'ichyvk', 'jenazu', 'nauwew', 'noimfc', 'bvvxnxy', 'cjemair', 'etqdcxt', 'hqwdqwy', 'thlmfrt', 'busivlqg', 'cfiypojm', 'dygpsqae', 'dzmqapfz', 'gzzhtrfz', 'ijikhyik', 'iwcejujv', 'jeviteai', 'wacbjbgu', 'jsnljcsbl', 'wynnqimrf', 'zajxxsoyl', 'lbwrppygrf', 'nceakmbixb', 'pkikkfxwlq', 'pouzguexyb', 'rxeneqraeg', 'scaqrxfnbl', 'slxybsnqjg', 'vdqrmlhazb', 'ypalccnbqb', 'cnwkpgoqybz', 'jmlmrywfhfx', 'jrsqrmtapse', 'kpulqqoowke', 'ldutizxiwad', 'ndvyrivxgdb', 'vbvjlifparc', 'dhjklzdazgpg', 'irgerzyfassi', 'reahnbgvkpro', 'ucokdsosmeeo', 'xinmxqjbweik', 'aaeuxpgyuoxcl', 'bhwcmrlyngjwa', 'ctavuaziyaafd', 'ddajvmfhjdpqv', 'drrslvcboezlc', 'hdpptoamcjgtr', 'kmqvqmzowbknv', 'liyqlbuxveadq', 'ydmtegpfhqiay', 'dcvlmlogruamud', 'dyzavdxmywmczn', 'edureokkyvvddv', 'fredpmyenviqdm', 'fznnqbfracwrsb', 'gyptnhcqtxfjwf', 'hhhemhumvpxgxo', 'ivngvcmibhedvo', 'nsxfyebfbywddn', 'ponrfhqorynrfe', 'pqhowqpnwzurse', 'stfwtfvprikmjl', 'udctpexupkbxdz', 'hgptibmszdbkaaf', 'rhcxvbggscymcyf', 'xkiowecbuawlwbt', 'yvefzsvpqbjqrlt', 'zmfvryuuvkzsfki']
    _, hsh = words_in(inwords)
    finished = True
    print("Output: ")
    print("\n".join(" ".join([i, str(lookup_word_count(i, hsh))]) for i in sorted(inwords, key=lambda x: test.l1Hash(x))))
    print("Done!")

