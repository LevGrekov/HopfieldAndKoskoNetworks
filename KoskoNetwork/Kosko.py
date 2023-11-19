import math
import time
import numpy as np


class KoskoNetwork:
    def __init__(self, dataX, dataY):
        self.__dataX = np.copy(dataX)
        self.__dataY = np.copy(dataY)

        self.__w = None
        self.learn()

    @staticmethod
    def activate(vector):
        return np.array([[-1 if i <= 0 else 1 for i in vector]]).T

    def learn(self):
        for i in range(len(self.__dataX)):
            if self.__w is None:
                self.__w = np.outer(self.__dataY[i].T, self.__dataX[i])
            else:
                self.__w += np.outer(self.__dataY[i].T, self.__dataX[i])

    @staticmethod
    def __is_in_res(vector, test_prev):
        test_prev.reverse()
        for i in test_prev:
            if np.array_equal(i, vector):
                return True
        return False

    @staticmethod
    def __is_in_set(vector, np_arr):
        for index, obr in enumerate(np_arr):
            if np.array_equal(obr, vector.flatten()):
                return index
        return None

    def recognize(self, vector):
        case = None
        if len(vector) == len(self.__dataY[0]):
            case = 0
        elif len(vector) == len(self.__dataX[0]):
            case = 1

        vector = np.array(vector)
        test_prev = []
        count = 0
        w = np.copy(self.__w)
        while not self.__is_in_res(vector, test_prev):
            test_prev.append(np.copy(vector))
            if case == 0:
                w = w.T
                vector = self.activate(np.dot(w, vector))
            elif case == 1:
                vector = self.activate(np.dot(w, vector))
                w = w.T
            count += 1

        if (self.__is_in_set(vector, self.__dataX) is not None) or (self.__is_in_set(vector, self.__dataY) is not None):
            print("Образ распознан")
        else:
            print("Образ не распознан")
        return vector

