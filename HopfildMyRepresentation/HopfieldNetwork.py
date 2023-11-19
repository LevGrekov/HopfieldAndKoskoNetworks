import time

import numpy as np
import DataLoader as dL


class HopfieldNetwork:
    def __init__(self, data):
        d = np.array(data)
        self.__data = np.copy(d)
        self.__w = None
        self.learn()

    @staticmethod
    def activate(vector):
        return np.array([-1 if i <= 0 else 1 for i in vector]).T

    @staticmethod
    def zero_out_diagonal(matrix):
        np.fill_diagonal(matrix, 0)
        return matrix

    def learn(self):
        for obr in self.__data:
            obr = np.array([obr])
            if self.__w is None:
                self.__w = np.dot(obr.T, obr)
            else:
                self.__w += np.dot(obr.T, obr)
            self.zero_out_diagonal(self.__w)

    def __isInTrainSet(self, vector):
        for index, obr in enumerate(self.__data):
            if np.array_equal(obr, vector):
                return index
        return None

    def recognize(self, vector):
        vector = np.array(vector)
        prev_vector = None
        count = 0
        while not np.array_equal(prev_vector, vector):
            count += 1
            idx = self.__isInTrainSet(vector)
            if idx is not None:
                print(f"Образ распознан. Похож на образ под номером {idx} из тренеровочного множества:")
                return idx
            else:
                prev_vector = vector
                vector = self.activate(np.dot(self.__w, vector))
                print(vector)
                print(prev_vector)
                print("----------------")
                time.sleep(1)
        print("Не удалось распознать образ. Конечный образ:")
        print(vector)
        return None
