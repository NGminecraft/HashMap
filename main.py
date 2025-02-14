import numpy as np
import threading
import warnings
import string
from random import randint, choice
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
        # This array holds a list of indices
        self.index_array = []
        self.function_array = []
        self.array = []
        self.size = 0
        self.supress_prints = False

        self.polyThread = None
        self.threadActive = False
        # Not really meant to be changed, for debugging
        self.l1Hash = lambda x: int("".join([str(ord(i.upper())) for i in list(x)]))

    def add(self, item):
        # Just to keep things clean, I'll define all the internal functions here
        
        if len(item) == 0:
            return

        try:
            a = self.__getitem__(item)
            a.number += 1
            return
        except IndexError:
            # adds an item onto the map, then starts the regression calculation seperately
            itemIndex = self.l1Hash(item)
            numberOfChars = len(item)

            if numberOfChars-1 < len(self.index_array):
                # The array already contains item(s) of the same length
                self.array.append(WordPair(item, itemIndex))
                self.index_array[numberOfChars-1].append(itemIndex)
                    
            elif numberOfChars-1 == len(self.index_array):
                # The array is the same size, so we can just add this to the end
                self.function_array.append(lambda v: numberOfChars-1)
                self.array.append(WordPair(item, itemIndex))
                self.index_array.append([itemIndex, ])
                    
            elif numberOfChars-1 > len(self.index_array):
                # We need to scale the array to add this to the right spot

                self.array.append(WordPair(item, itemIndex))
                self.function_array.extend([None]*(numberOfChars-1-len(self.function_array)))
                for i in range(numberOfChars-1-len(self.index_array)):
                    self.index_array.append([])
                self.index_array.append([itemIndex, ])
                self.function_array.append(lambda v: numberOfChars-1)

            # Let's run this on a seperate thread, this could get time consuming, and any time saved is something
            self.array.sort(key=lambda x: self.l1Hash(x.word))
            self.assert_safe()
            self.polyThread = threading.Thread(target=self.calculate_regression, args=(self.index_array, ))
            self.polyThread.start()
            self.threadActive = True


    def calculate_regression(self, lst):
        """ This function takes a list and then uses numpy to find a polynomial function that closely models the list
        The X axis is the values passed in, the Y axis are the indices from 0, len(lst)
        """

        # Since every word of a character count needs to have its own function
        for i, v in enumerate(lst):
            if len(v) == 0:
                continue
            offset = sum([len(xs) for xs in lst[:i]])
            # TODO: Offset the values to compensate for items in smaller word lengths
            if len(v) == 1:
                # Regression doesn't work with one value, so just skip it
                self.poly = lambda x: offset
                return
            
            # It doesn't strictly have to be sorted, but less work on the polyfit function
            v.sort()
            array = np.array(v).astype(float)
            scaled_array = np.log10(array)
            xs = np.array([j for j in range(len(v))]) 
            max_error = 0
            pw = 1
            while True:
                # We use NumPY to generate a quadratic regression to fit the data
                # We loop through until we find the first power polynomial to
                # meet the criteria (close enough to needed value)
                try:
                    coefficents = np.polyfit(scaled_array, xs, pw)
                except np.linalg.LinAlgError:
                    print(lst)
                coefficents[-1] += offset # Offset the intercept to match correct index
                polynomial = np.poly1d(coefficents)
                for j, v in enumerate(scaled_array.tolist()):
                    num = polynomial(v)
                    max_error = max(max_error, abs(num - xs[j] - offset))
                    if not round(num) == xs[j] + offset:
                        break
                    
                else:
                    self.function_array[i] = polynomial
                    break
                pw += 1
    
    def assert_safe(self):
        """ This method checks to make sure that the regression function finished"""
        if self.threadActive:
            if not self.supress_prints:
                print("Waiting for regression thread to finish")
            time = time_ns()
            self.polyThread.join()
            self.threadActive = False
            if not self.supress_prints:
                print(f"Thread finished in {(time_ns() - time)} nanoseconds")

    def __getitem__(self, index):
        scaled_index = log(self.l1Hash(index), 10)
        self.assert_safe()
        if len(self.array) == 0:
            raise IndexError("The Hash Map is empty")
        else:
            # We take the base index, that will likely never have collisions, then run it through our approximated function
            # We calculated in the calculate_regression function
            # We need to check to make sure the item we got is the same as the one we request
            funct = self.function_array[len(index)-1]
            if funct is None:
                raise IndexError("The item is not in the Hash Map")
            
            value_to_check = round(funct(scaled_index))
            if value_to_check-1 > len(self.array):
                raise IndexError("The item is not in the Hash Map")
            item = self.array[round(funct(scaled_index))]
            if item.key == self.l1Hash(index):
                return item
            else:
                raise IndexError("The item is not in the Hash Map, or there was a mismatch")

test_hash = HashMap()

def words_in(words):
    words.sort(key=lambda x: a.l1Hash(x)) # This may help smooth out the data and avoid bumps
    print([a.l1Hash(i) for i in words])
    for i in words:
        test_hash.add(i)
    return len(test_hash.array), 0

def lookup_word_count(word):
    return test_hash[word].number

if __name__ == "__main__":
    a = HashMap()
    a.add("a")
    a.add("a")
    a.add("b")
    print(a["a"].number)

    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince", "raspberry", "strawberry", "tangerine", "ugli", "vanilla", "watermelon", "xigua", "yellowfruit", "zucchini"]

    print(len(words))

    print(words_in(words))
    print(lookup_word_count("apple"))
    
    def generate_random_word():
        return ''.join([choice(string.ascii_lowercase) for _ in range(randint(1, 15))])
        
    words_in([generate_random_word() for _ in range(100)])
