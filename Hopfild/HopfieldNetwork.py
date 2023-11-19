import numpy as np


class HopfieldNetwork:

    def learn(self, data):
        d = np.array(data)
        self.__data = np.copy(d)
        W = None
        for obr in d:
            obr = np.array([obr])
            if W is None:
                W = np.dot(obr.T, obr)
            else:
                W += np.dot(obr.T, obr)
            for i in range(len(W)):
                W[i][i] = 0
        self.__W = W

    def activate(self, t):
        return np.array([
            [-1 if i[0] <= 0 else 1 for i in t]
        ]).T

    def recognize(self, data):
        i = 0
        res = []
        for test in data:
            test = np.array([test]).T
            test_prev = []
            k = 0
            while not self.__in_res(test, test_prev):
                test_prev.append(np.copy(test))
                test = self.activate(np.dot(self.__W, test))
                k += 1
            print(k)

            res.append(None)
            j = 0
            for obr in self.__data:
                if np.array_equal(obr, test.T[0]):
                    res[i] = j
                    break
                j += 1
            i += 1
        return res

    def __in_res(self, test, test_prev):
        test_prev.reverse()
        for obr in test_prev:
            if (np.array_equal(obr, test)): return True
        return False
