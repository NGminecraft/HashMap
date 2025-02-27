import numpy as np
import threading
import warnings
import string
from random import randint, choice
from time import time_ns
from math import log
from concurrent import futures
from re import sub
import torch
import torch.nn as nn
import torch.optim as optim

class funct_model(nn.Module):
    def __init__(self):
        super(funct_model, self) .__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)
    


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
        self.array = []
        self.size = 0

        self.return_zero = False

        self.indices = []
        self.indices_min = None
        self.indices_range = 0

        self.polyThread = None
        self.threadActive = False
        # Not really meant to be changed, for debugging
        self.l1Hash = lambda x: int("".join([str(ord(i.upper())) for i in list(x)]))
        
        self.criterion = nn.MSELoss()
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
        """
        self.polyThread = threading.Thread(target=self.calculate_regression)
        self.polyThread.start()
        self.threadActive = True
        """
        self.calculate_regression()

    def calculate_regression(self):
        """ This function takes a list and then uses numpy to find a polynomial function that closely models the list
        The X axis is the values passed in, the Y axis are the indices from 0, len(lst)
        """
        
        self.model = funct_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.01)
        
        indices_normalized = [(i-self.indices_min)/self.indices_range for i in self.indices]

        expected_indices = range(len(self.array))

        MAX_ERROR = 0.4 # Maximum error
        TOLERANCE = 5 # Number of times after it's first pass it needs to succeed
        tolerance_epochs = 0

        in_indices_t = torch.tensor(indices_normalized, dtype=torch.float32).view(-1, 1)
        exp_indices_t = torch.tensor(expected_indices, dtype=torch.float32).view(-1)

        while True:
            self.optimizer.zero_grad()
            in_predicted = self.model(in_indices_t)
            loss = self.criterion(in_predicted, exp_indices_t)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                error = torch.max(torch.abs(in_predicted - exp_indices_t)).item()
            print(error)

            if error < MAX_ERROR:
                tolerance_epochs += 1
                if tolerance_epochs >= TOLERANCE:
                    break
            else:
                tolerance_epochs += 1

        

    def soft_insert(self, item, count=1):
        item = sub("[^A-Za-z]", "", item)
        if len(item) == 0:
            return
        
        item_index = self.l1Hash(item)

        self.array.append(WordPair(item, item_index, count))
        self.indices.append(item_index)
        if not self.indices_min:
            self.indices_min = item_index
        elif item_index < self.indices_min and len(self.array) != 1:
            self.return_zero = False
            self.indices_min = item_index
        elif len(self.array) == 1:
            self.return_zero = True
        self.indices_range = max(self.indices) - self.indices_min

     
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
            print("Waiting for regression thread to finish")
            time = time_ns()
            self.polyThread.join()
            self.threadActive = False
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
            
            if self.return_zero:
                if index == self.array[0].word:
                    return self.array[0], 1
                else:
                    raise IndexError("Item not in list")

            result = self.model(torch.tensor([scaled_index], dtype=torch.float32).view(1, 1))
            

            _, predicted_class = torch.max(result, dim=1)
            predicted_class = predicted_class.item()

            if result is None:
                raise IndexError("The item is not in the array")

            item = self.array[predicted_class]
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
    inwords = [generate_random_word() for _ in range(100)]
    #inwords = ['g', 'n', 'n', 'hp', 'ij', 'md', 'ra', 'so', 'ty', 'xu', 'drd', 'fhj', 'gyg', 'hih', 'mfk', 'pae', 'umc', 'xfk', 'zee', 'cxou', 'iwld', 'pdiw', 'zovk', 'clyeb', 'efjsw', 'gxvwc', 'wjoaa', 'yxxut', 'zxrnn', 'etmlxo', 'fthzoy', 'ichyvk', 'jenazu', 'nauwew', 'noimfc', 'bvvxnxy', 'cjemair', 'etqdcxt', 'hqwdqwy', 'thlmfrt', 'busivlqg', 'cfiypojm', 'dygpsqae', 'dzmqapfz', 'gzzhtrfz', 'ijikhyik', 'iwcejujv', 'jeviteai', 'wacbjbgu', 'jsnljcsbl', 'wynnqimrf', 'zajxxsoyl', 'lbwrppygrf', 'nceakmbixb', 'pkikkfxwlq', 'pouzguexyb', 'rxeneqraeg', 'scaqrxfnbl', 'slxybsnqjg', 'vdqrmlhazb', 'ypalccnbqb', 'cnwkpgoqybz', 'jmlmrywfhfx', 'jrsqrmtapse', 'kpulqqoowke', 'ldutizxiwad', 'ndvyrivxgdb', 'vbvjlifparc', 'dhjklzdazgpg', 'irgerzyfassi', 'reahnbgvkpro', 'ucokdsosmeeo', 'xinmxqjbweik', 'aaeuxpgyuoxcl', 'bhwcmrlyngjwa', 'ctavuaziyaafd', 'ddajvmfhjdpqv', 'drrslvcboezlc', 'hdpptoamcjgtr', 'kmqvqmzowbknv', 'liyqlbuxveadq', 'ydmtegpfhqiay', 'dcvlmlogruamud', 'dyzavdxmywmczn', 'edureokkyvvddv', 'fredpmyenviqdm', 'fznnqbfracwrsb', 'gyptnhcqtxfjwf', 'hhhemhumvpxgxo', 'ivngvcmibhedvo', 'nsxfyebfbywddn', 'ponrfhqorynrfe', 'pqhowqpnwzurse', 'stfwtfvprikmjl', 'udctpexupkbxdz', 'hgptibmszdbkaaf', 'rhcxvbggscymcyf', 'xkiowecbuawlwbt', 'yvefzsvpqbjqrlt', 'zmfvryuuvkzsfki']
    words_in(inwords)
    finished = True
    print("Output: ")
    print("\n".join(" ".join([i, str(lookup_word_count(i))]) for i in sorted(inwords, key=lambda x: test_hash.l1Hash(x))))
    print("Done!")
    
