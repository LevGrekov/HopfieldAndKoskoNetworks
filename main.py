# from Hopfild.DataLoader import DataLoader
# from Hopfild.HopfieldNetwork import HopfieldNetwork as hopnn
#
# print("Hopfild:")
# d = DataLoader("Hopfild/data.xlsx")
# data = d.get_data()
# h = hopnn()
# h.learn(data)
# td = DataLoader("Hopfild/test.xlsx")
# test = td.get_data()
# res = h.recognize(test)
# print(res)
# from DataLoader import DataLoader
# from HopfieldNetwork import HopfieldNetwork
# tst_d = DataLoader("test.xlsx")
# test = tst_d.get_data()
#
# d = DataLoader("data.xlsx")
# tr = d.get_data()
# h = HopfieldNetwork(tr)
#
# h.recognize(test[0])
# # h = HopfieldNetwork([[-1,1,-1,1],[1,-1,1,1],[-1,1,-1,-1]])
# # a = h.recognize([-1,-1,-1,-1])
# # print(a)

import DataLoader as dl
from KoskoNetwork.Kosko import KoskoNetwork
dataX = [[-1,1,-1,1,1,1,1,-1],[1,-1,1,1,1,1,1,1],[1,-1,-1,-1,1,1,-1,1]]
dataY = dl.DataLoader.get_associated_set(dataX)

h = KoskoNetwork(dataX, dataY)
a = h.recognize([-1,1,-1,-1,1,1,1,-1])
print(a, "\n")
b = h.recognize([-1, -1, -1, -1])
print(b)
