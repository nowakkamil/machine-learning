# pylint: disable=unbalanced-tuple-unpacking

from knn import KNN
from numpy import array, array_split
import pandas as pd

training_data = array(pd.read_csv("iris.data.learning", header=None))
knn = KNN(training_data)

new_data, labels = array_split(array(pd.read_csv("iris.data.test", header=None)), [4], axis=1)
for i in range(30, 1, -1):
    knn.change_k(i)
    print("k = {} => ".format(str(i).rjust(2, ' ')), end="")
    print(f'{knn.score(new_data, labels):3.2f}')

