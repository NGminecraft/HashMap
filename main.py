import numpy as np
import threading
import warnings
from time import time_ns
from math import log


class WordPair:
    def __init__(self, word, key, number=1):
        self.word = word
        self.number = number
        self.key = key


class HashMap:
    def __init__(self):
        warnings.simplefilter('ignore', np.RankWarning) # These are obnoxious
        # This array holds all the indices, so we don't have to keep calculating them
        self.index_array = []
        self.array = []
        self.size = 0

        self.polyThread = None
        self.threadActive = False
        self.poly = None # If this is None, hash is empty
        # Not really meant to be changed, for debugging
        self.l1Hash = lambda x: int("".join([str(ord(i.upper())) for i in list(x)]))

    def add(self, item):
        try:
            a = self.__getitem__(item)
            a.number += 1
            return
        except IndexError:
            # adds an item onto the map, then starts the regression calculation seperately
            itemIndex = self.l1Hash(item)
            #We build our indexing around this array, so we can just stick it at the end, and the
            #Regression function will handle the rest
            self.array.append(WordPair(item, itemIndex))
            self.index_array.append(itemIndex)
            self.calculate_regression(self.index_array)
            # Let's run this on a seperate thread, this could get time consuming, and any time saved is something
            """self.assert_safe()
            self.polyThread = threading.Thread(target=self.calculate_regression, args=(self.index_array, ))
            self.polyThread.start()
            self.threadActive = True"""


    def calculate_regression(self, lst):
        """ This function takes a list and then uses numpy to find a polynomial function that closely models the list
        The X axis is the values passed in, the Y axis are the indices from 0, len(lst)"""
        if len(lst) == 1:
            # Regression doesn't work with one value, so just skip it
            self.poly = lambda x: 0
            return
        # It doesn't strictly have to be sorted, but less work on the polyfit function
        lst.sort()
        array = np.array(lst).astype(float)
        scaled_array = np.log10(array)
        xs = np.array([i for i in range(len(lst))]) 
        assert not np.isnan(scaled_array).any() and not np.isinf(scaled_array).any() 
        print(scaled_array)
        pw = 1
        while True:
            # We use NumPY to generate a quadratic regression to fit the data
            # We loop through until we find the first power polynomial to
            # meet the criteria (close enough to needed value)
            coefficents = np.polyfit(scaled_array, xs, pw)
            polynomial = np.poly1d(coefficents)
            for i, v in enumerate(scaled_array.tolist()):
                num = polynomial(v)
                if not round(num) == xs[i]:
                    break
                
            else:
                self.poly = polynomial
                break
            pw += 1
    
    def assert_safe(self):
        """ This method checks to make sure that the regression function finished"""
        if self.threadActive:
            print("Waiting for regression thread to finish")
            time = time_ns()
            self.polyThread.join()
            self.threadActive = False
            print(f"Thread finished in {(time_ns() - time)} nanoseconds")

    def __getitem__(self, index):
        scaled_index = log(self.l1Hash(index), 10)
        self.assert_safe()
        if not self.poly:
            raise IndexError
        else:
            # We take the base index, that will likely never have collisions, then run it through our approximated function
            # We calculated in the calculate_regression function
            # We need to check to make sure the item we got is the same as the one we request
            item = self.array[round(self.poly(scaled_index))]
            if item.key == self.l1Hash(index):
                return item
            else:
                raise IndexError

def words_in(words):
    a = HashMap()
    for i in words:
        a.add(i)
    return len(a.array), 0

if __name__ == "__main__":
    a = HashMap()
    a.add("a")
    a.add("a")
    a.add("b")
    print(a["a"].number)

    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince", "raspberry", "strawberry", "tangerine", "ugli", "vanilla", "watermelon", "xigua", "yellowfruit", "zucchini"]

    print(words_in(words))
