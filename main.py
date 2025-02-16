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
#        warnings.simplefilter('ignore', np.RankWarning) # These are obnoxious
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

        def format_array(lst):
            lst.sort()
            arr = np.array(lst).astype(np.double)
            return np.log10(arr)

        def create_polynomial(unformatted_indices, expected_indices):
            indices = format_array(unformatted_indices)
            power = 1
            while power <= 100:
                try:
                    coes = np.polyfit(indices, expected_indices, power)
                    polynomial =  np.poly1d(coes)
                except np.RankWarning:
                    power += 1
                else:
                    # Ok now we need to check and make sure the polynomial actually fits well enough
                    for each_key, expected_output in zip(indices, expected_indices):
                        returned = polynomial(each_key)
                        # If any fail, we know that we need to refit
                        if round(returned) != expected_output:
                            power+=1
                            break
                    else:
                        # They all passed, so this polynomial works
                        return polynomial
            else:
                # If we get here, that's bad. We couldn't get a polynomial to reliably work
                # Further data processing is needed.

                # Probably repeat the same algorithm, but chunk the words by the first digit of the index, instead of just word legnth

                idx_start = 0
                current_first_char = str(unformatted_indices[0])[0]
                secondary_function_list = []
                first_digits = [int(current_first_char)]

                for i in range(1,len(unformatted_indices)):
                    if str(unformatted_indices[i])[0] != current_first_char:
                        func = create_polynomial(unformatted_indices[idx_start:i], expected_indices[idx_start:i])
                        secondary_function_list.append(func)

                        idx_start = i
                        current_first_char = str(unformatted_indices[i])[0]
                        first_digits.append(int(current_first_char))
                
                final_function_list = [None for _ in range(first_digits[-1]-5)]
                for d, f in zip(first_digits, secondary_function_list):
                    final_function_list[d-6] = f


                def returned_function(x):
                    main_idx = 10**x
                    first_digit = int(str(main_idx)[0])
                    function = final_function_list[first_digit-6]
                    return function(x)
                
                return returned_function
        

        # lst is a list of lists. Each item in the list is a list of each words hash, split by word size
        # The get will go into the function array at the equivalent index array index, and then call it
        for main_list_index, secondary_list in enumerate(lst):
            # The indexing in the primary array is the offset (all the items in the lists before this one) + the result of the function
            offset = sum(len(i) for i in lst[:main_list_index])
            if len(secondary_list) == 0:
                self.function_array[main_list_index] = None
                continue
            elif len(secondary_list) == 1:
                self.function_array[main_list_index] = lambda x: offset # If there is only one item, we can just return the index
                continue
            expected_indices = [i for i in range(offset, offset+len(secondary_list))] # This forgoes needing to manually add, or change the polynomial
            poly = create_polynomial(secondary_list, expected_indices)

            self.function_array[main_list_index] = poly
            
    
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
        
    inwords = [generate_random_word() for _ in range(100)]
    words_in(inwords)
    print("Output: ")
    print("\n".join(lookup_word_count(i) for i in inwords))
    print("Done!")
