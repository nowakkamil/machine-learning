# pylint: disable=unbalanced-tuple-unpacking

from knn import KNN
from numpy import array, array_split
from custom_exceptions import *
import pandas as pd
import unittest

class TestKNNAlgorithm(unittest.TestCase):
    def test_perform_calculations(self):
        try:
            with open("iris.data.learning", 'r') as read_file:
                training_data_file = pd.read_csv(read_file, header=None)
                training_data = array(training_data_file)
                knn = KNN(training_data)

            with open("iris.data.test", 'r') as read_file:
                data_file = pd.read_csv(read_file, header=None)
                data = array(data_file)
                new_data, labels = array_split(data, [4], axis=1)

            for i in range(30, 1, -1):
                knn.change_k(i)
                print("k = {} => ".format(str(i).rjust(2, ' ')), end="")
                print(f'{knn.score(new_data, labels):1.3f}')
        except Exception:
            self.fail("test_perform_calculations function raised Exception")
        
    def test_exception_will_be_thrown_when_negative_k_is_passed(self):
        k = -1
        self.assertRaises(InvalidValueOfArgumentException, KNN.validate_parameters, k)

    def test_exception_will_be_thrown_when_zero_k_is_passed(self):
        k = 0
        self.assertRaises(InvalidValueOfArgumentException, KNN.validate_parameters, k)

    def test_exception_will_be_thrown_when_arrays_are_of_different_sizes(self):
        first_array = 1, 2, 3
        second_array = 4, 5
        self.assertRaises(ArgumentsNotEqualException, KNN.validate_equal_length_of_containers, first_array, second_array)

    def test_exception_will_be_thrown_when_one_of_the_arrays_is_empty(self):
        first_array = []
        second_array = 1
        self.assertRaises(EmptyContainerException, KNN.validate_equal_length_of_containers, first_array, second_array)

if __name__ == '__main__':
    unittest.main()
