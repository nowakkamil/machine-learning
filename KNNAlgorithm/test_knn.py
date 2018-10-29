# pylint: disable=unbalanced-tuple-unpacking

from knn import KNN
from numpy import array, array_split
from ddt import ddt, data, unpack
from custom_exceptions import *
import pandas as pd
import sys
import unittest


@ddt
class TestKNNAlgorithm(unittest.TestCase):

    def setUp(self):
        try:
            with open("iris.data.learning", 'r') as read_file:
                training_data_file = pd.read_csv(read_file, header=None)
                training_data = array(training_data_file)
                self.knn = KNN(training_data)
        except Exception:
            self.fail("setUp() raised Exception")

    def test_perform_calculations(self):
        try:
            with open("iris.data.test", 'r') as read_file:
                data_file = pd.read_csv(read_file, header=None)
                data = array(data_file)
                new_data, labels = array_split(data, [4], axis=1)

            for i in range(30, 1, -1):
                self.knn.change_k(i)
                print("k = {} => ".format(str(i).rjust(2, ' ')), end="")
                print(f'{self.knn.score(new_data, labels):1.3f}')
        except Exception:
            self.fail("test_perform_calculations() raised Exception")

    @data(-sys.maxsize, -1, 0)
    def test_exception_will_be_thrown_when_invalid_k_is_passed(self, k):
        self.assertRaises(InvalidValueOfArgumentException,
                          KNN.validate_parameter, k)

    @data(([1, 2], [3, 4, 5]))
    @unpack
    def test_exception_will_be_thrown_when_containers_are_of_different_sizes(
        self, first_container, second_container
    ):
        self.assertRaises(ArgumentsNotEqualException, KNN.validate_equal_length_of_containers,
                          first_container, second_container)

    @data(([], [3]))
    @unpack
    def test_exception_will_be_thrown_when_one_of_the_containers_is_empty(
        self, first_container, second_container
    ):
        self.assertRaises(EmptyContainerException, KNN.validate_equal_length_of_containers,
                          first_container, second_container)


if __name__ == '__main__':
    unittest.main()
