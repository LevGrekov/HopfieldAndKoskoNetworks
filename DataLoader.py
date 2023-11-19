import math

import numpy as np
import openpyxl


def change(x):
    if x == 1:
        return 1
    else:
        return -1


class DataLoader:
    def __init__(self, filename="data.xlsx", data_size=(6, 3)):
        wb = openpyxl.load_workbook(filename)
        sh = wb['data']
        max_r = sh.max_row
        full_size = data_size[0] * data_size[1]
        self.__train_set = [
            [
                1 if sh[chr(i % data_size[1] + ord('A')) + str(i // data_size[1] + 1)].value == 1 else -1
                for i in range(k * full_size, (k + 1) * full_size)
            ] for k in range(max_r // data_size[0])
        ]

    def get_data(self):
        return self.__train_set.copy()

    @staticmethod
    def get_associated_set(dataX):
        max_l = 2 ** math.ceil(math.log2(len(dataX)))
        dataY = np.zeros((len(dataX), max_l), dtype=int)
        for i in range(len(dataX)):
            dataY[i] = DataLoader.to_binary_bipolar_representation(i, max_l)
        return dataY

    @staticmethod
    def to_binary_bipolar_representation(i, max_l):
        val = f"{bin(i + 1)[2:][::-1]:0{max_l}}"[::-1]
        return np.array(list(map(lambda i: -1 if i == '0' else 1, val)))