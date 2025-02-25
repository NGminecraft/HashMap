import numpy as np
import threading
import warnings
import string
from random import randint, choice
from time import time_ns
from math import log
from concurrent import futures
from re import sub


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
        self.function_array = []
        self.funct = None
        self.array = []
        self.size = 0
        self.supress_prints = False

        self.polyThread = None
        self.threadActive = False
        # Not really meant to be changed, for debugging
        self.l1Hash = lambda x: int("".join([str(ord(i.upper())) for i in list(x)]))
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
        self.polyThread = threading.Thread(target=self.calculate_regression, args=(self.index_array, ))
        self.polyThread.start()
        self.threadActive = True

    def calculate_regression(self, lst):
        """ This function takes a list and then uses numpy to find a polynomial function that closely models the list
        The X axis is the values passed in, the Y axis are the indices from 0, len(lst)
        """

        all_indices = []
        for i in lst:
            all_indices.extend(i)

        conditions = []
        functs = []
        x_vals = [] # We can just sort this later and evetrything will be correct (presumably)

        def format_array(lst):
            lst.sort()
            arr = np.array(lst).astype(np.double)
            return arr

        def create_polynomial(unformatted_indices, expected_indices, backup_letters=0):
            """Takes numbers, and what they need to be mapped too then goes and makes a function for it"""
            indices = format_array(unformatted_indices)
            match len(unformatted_indices):
                case 1:
                     # If there is only one item, return it's index
                    return [(unformatted_indices[0], unformatted_indices[-1])], lambda x: expected_indices[0]
                case 2:
                    
                    # I can actually do this, and this is probably a bit faster
                    slope = (expected_indices[1]- expected_indices[0])/(indices[1]-indices[0])

                    intercept = expected_indices[0] - (slope * indices[0])
                    
                    def linear_function(x, m=slope, b=intercept):
                        return m * x + b
                    return [(unformatted_indices[0], unformatted_indices[-1])], [linear_function, ]
                case _:
                    power = 0
                    MAX_POWER = 10
                    while power <= MAX_POWER:
                        coes = np.polyfit(indices, expected_indices, power)
                        polynomial =  np.poly1d(coes)
                        # Ok now we need to check and make sure the polynomial actually fits well enough
                        for each_key, expected_output in zip(indices, expected_indices):
                            returned = polynomial(each_key)
                            # If any fail, we know that we need to refit
                            error = abs(returned-expected_output)
                            if error > 0.49: # I originally just used round(returned) != expected_output, but was running into what I think is floating point error
                                power+=1
                                break # TODO, allow functions to map to only part of the data
                        else:
                            # They all passed, so this polynomial works
                            return [(unformatted_indices[0], unformatted_indices[-1])], [polynomial, ]
                    else:
                        # If we get here, that's bad (we get here quite often). We couldn't get a polynomial to reliably work
                        # Further data processing is needed.

                        # Probably repeat the same algorithm, but chunk the words by the first digit of the index, instead of just word legnth

                        idx_start = 0
                        current_first_char = str(unformatted_indices[0])[backup_letters]
                        secondary_function_list = []
                        first_digits = [int(current_first_char)]

                        locations = []


                        for i in range(1,len(unformatted_indices)):
                            if str(unformatted_indices[i])[backup_letters] != current_first_char:
                                loc, func = create_polynomial(unformatted_indices[idx_start:i], expected_indices[idx_start:i], backup_letters+1)
                                secondary_function_list.append(func)
                                locations.append(loc)

                                idx_start = i
                                current_first_char = str(unformatted_indices[i])[backup_letters]
                                first_digits.append(int(current_first_char))
                        else:
                            loc, func = create_polynomial(unformatted_indices[idx_start:], expected_indices[idx_start:], backup_letters+1)
                            secondary_function_list.append(func)
                            locations.append(loc)
                        
                        """
                        if backup_letters%2 == 0 and first_digits[0] >= 6:
                            normalizer = -6 # Only the first digit of each ascii letter can be shorted
                            # It starts at A, 65, but we can have a 70, and thus can't minus 6 if were looking at the second digit
                        else:
                            normalizer = 0

                        final_function_list = [None for _ in range(first_digits[-1]+normalizer+1)]
                        for d, f in zip(first_digits, secondary_function_list):
                            final_function_list[d+normalizer] = f

                
                        def returned_function(x, f_list=final_function_list, num_digits=backup_letters, norm=normalizer):
                            main_idx = round(10**x)
                            first_digit = int(str(main_idx)[num_digits])
                            funct = f_list[first_digit+norm]
                            funct_result = funct(x)
                            if type(funct_result) is tuple:
                                value = funct_result[0]
                                steps = funct_result[1] + 1
                            else:
                                value = funct_result
                                steps = 1
                            return value, steps
                        """
                        return locations, secondary_function_list

        def single_instance(idx, lst_to_use, offset):
            """This function uses create_polynomial to map functions to the correct spots
            It's seperate to allow multithreading
            """
            # The indexing in the primary array is the offset (all the items in the lists before this one) + the result of the function
            
            x_vals.extend(lst_to_use)

            match len(lst_to_use):
                case 0:
                    pass

                case 1:
                    conditions.append((lst_to_use[0], lst_to_use[-1]))
                    functs.append(lambda x: offset) # Just return the offset if everything works
                case _:
                    expected_indices = [i for i in range(offset, offset+len(lst_to_use))] # This forgoes needing to manually add, or change the polynomial
                    
                    conds, polys = create_polynomial(lst_to_use, expected_indices)

                    conditions.extend(conds)
                    functs.extend(polys)


        # lst is a list of lists. Each item in the list is a list of each words hash, split by word size
        # The get will go into the function array at the equivalent index array index, and then call it
        total = 0
        functions = []
        with futures.ThreadPoolExecutor() as executor:
            for i, v in enumerate(lst):
                functions.append(executor.submit(single_instance, i, v, total))
                total += len(v)
        
        for i in functions:
            i.result()

        x_vals.sort()

        for i in range(len(conditions)):
            conditions[i] = lambda x: (conditions[i][0] < x < conditions[i][1])

        assert len(conditions) == len(functs)

        self.funct = np.piecewise(all_indices, conditions, functs, default=None)

    def soft_insert(self, item, count=1):
        item = sub("[^A-Za-z]", "", item)
        if len(item) == 0:
            return
        
        itemIndex = self.l1Hash(item)
        numberOfChars = len(item)
        
        if numberOfChars-1 < len(self.index_array):
            # The array already contains item(s) of the same length
            self.array.append(WordPair(item, itemIndex, count))
            self.index_array[numberOfChars-1].append(itemIndex)
                
        elif numberOfChars-1 == len(self.index_array):
            # The array is the same size, so we can just add this to the end
            self.function_array.append(lambda v: numberOfChars-1)
            self.array.append(WordPair(item, itemIndex, count))
            self.index_array.append([itemIndex, ])
                
        elif numberOfChars-1 > len(self.index_array):
            # We need to scale the array to add this to the right spot

            self.array.append(WordPair(item, itemIndex, count))
            self.function_array.extend([None]*(numberOfChars-1-len(self.function_array)))
            for i in range(numberOfChars-1-len(self.index_array)):
                self.index_array.append([])
            self.index_array.append([itemIndex, ])
            self.function_array.append(lambda v: numberOfChars-1)
     
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
        scaled_index = self.l1Hash(index)
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

test_hash = HashMap()
finished = False

def words_in(words):
    words.sort()
    return test_hash.words_in(words)

def lookup_word_count(word):
    item = test_hash[word]
    return item[0].number, item[1]

if __name__ == "__main__":
    def generate_random_word():
        return ''.join([choice(string.ascii_lowercase) for _ in range(randint(1, 15))])
        
    
    #inwords = ["a", "a", "as", "at"]
    inwords = [generate_random_word() for _ in range(10)]
    #inwords = ['g', 'n', 'n', 'hp', 'ij', 'md', 'ra', 'so', 'ty', 'xu', 'drd', 'fhj', 'gyg', 'hih', 'mfk', 'pae', 'umc', 'xfk', 'zee', 'cxou', 'iwld', 'pdiw', 'zovk', 'clyeb', 'efjsw', 'gxvwc', 'wjoaa', 'yxxut', 'zxrnn', 'etmlxo', 'fthzoy', 'ichyvk', 'jenazu', 'nauwew', 'noimfc', 'bvvxnxy', 'cjemair', 'etqdcxt', 'hqwdqwy', 'thlmfrt', 'busivlqg', 'cfiypojm', 'dygpsqae', 'dzmqapfz', 'gzzhtrfz', 'ijikhyik', 'iwcejujv', 'jeviteai', 'wacbjbgu', 'jsnljcsbl', 'wynnqimrf', 'zajxxsoyl', 'lbwrppygrf', 'nceakmbixb', 'pkikkfxwlq', 'pouzguexyb', 'rxeneqraeg', 'scaqrxfnbl', 'slxybsnqjg', 'vdqrmlhazb', 'ypalccnbqb', 'cnwkpgoqybz', 'jmlmrywfhfx', 'jrsqrmtapse', 'kpulqqoowke', 'ldutizxiwad', 'ndvyrivxgdb', 'vbvjlifparc', 'dhjklzdazgpg', 'irgerzyfassi', 'reahnbgvkpro', 'ucokdsosmeeo', 'xinmxqjbweik', 'aaeuxpgyuoxcl', 'bhwcmrlyngjwa', 'ctavuaziyaafd', 'ddajvmfhjdpqv', 'drrslvcboezlc', 'hdpptoamcjgtr', 'kmqvqmzowbknv', 'liyqlbuxveadq', 'ydmtegpfhqiay', 'dcvlmlogruamud', 'dyzavdxmywmczn', 'edureokkyvvddv', 'fredpmyenviqdm', 'fznnqbfracwrsb', 'gyptnhcqtxfjwf', 'hhhemhumvpxgxo', 'ivngvcmibhedvo', 'nsxfyebfbywddn', 'ponrfhqorynrfe', 'pqhowqpnwzurse', 'stfwtfvprikmjl', 'udctpexupkbxdz', 'hgptibmszdbkaaf', 'rhcxvbggscymcyf', 'xkiowecbuawlwbt', 'yvefzsvpqbjqrlt', 'zmfvryuuvkzsfki']
    words_in(inwords)
    finished = True
    print("Output: ")
    print("\n".join(" ".join([i, str(lookup_word_count(i))]) for i in sorted(inwords, key=lambda x: test_hash.l1Hash(x))))
    print("Done!")
    
