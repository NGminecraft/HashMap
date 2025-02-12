import numpy as np
import threading


class HashMap:
    def __init__(self):
        self.array = []
        self.size = 0

        self.polyThread = None
        self.threadActive = False
        self.poly = None # If this is None, hash is empty
        # Not really meant to be changed, for debugging
        self.l1Hash = lambda x: int("".join([str(ord(i)) for i in list(x)]))

    def add(self, item):
        # adds an item onto the map, then starts the regression calculation seperately
        item = self.l1Hash(item)
        if self.threadActive:
            self.polyThread.join()
        self.array.append(item)
        print(self.array)
        self.polyThread = threading.Thread(target=self.calculate_regression, args=self.array)
        self.polyThread.start()
        self.threadActive = True


    def calculate_regression(self, lst):
        """ This function takes a list and then uses numpy to find a polynomial function that closely models the list
        The X axis is the values passed in, the Y axis are the indices from 0, len(lst)"""
        if type(lst) is int:
            lst = [lst]
        lst.sort()
        array = np.array(lst)
        xs = np.array([i for i in range(len(lst))]) 
        pw = 1
        while True:
            coefficents = np.polyfit(array, xs, pw)
            polynomial = np.poly1d(coefficents)
            for i, v in enumerate(lst):
                num = polynomial(v)
                print(num, xs[i], pw)
                if not round(num) == xs[i]:
                    break
                
            else:
                self.poly = polynomial
                break
            pw += 1

    def __getitem__(self, index):
        pass

a = HashMap()
a.add("a")
