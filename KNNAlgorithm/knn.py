# pylint: disable=unbalanced-tuple-unpacking

from numpy import array_split, squeeze
from scipy.spatial import distance
from sklearn import preprocessing
from custom_exceptions import *
import operator as o
import os


class Sample():

    def __init__(self, dist, label):
        self.dist = dist
        self.label = label

    def __str__(self):
        return "label: " + self.label + ", dist: " + str(self.dist)


class KNN():

    euclidean = distance.euclidean
    manhattan = distance.cityblock

    def __init__(self, training_data, k=3, metrics=euclidean):
        try:
            KNN.validate_parameters(k)
        except InvalidValueOfArgumentException as e:
            print(e.message)
            os._exit(1)

        shape = training_data.shape
        features = shape[1] - 1
        self.data, self.target = array_split(training_data, [features], axis=1)
        self.data = self.data.astype(float)
        self.std_sc = preprocessing.StandardScaler()
        self.std_sc.fit(self.data)
        self.data = self.std_sc.transform(self.data)

        self.target = list(squeeze(self.target))
        self.k = k
        self.metrics = metrics

    @staticmethod
    def validate_parameters(k):
        if k <= 0:
            raise InvalidValueOfArgumentException(
                "Value of k should be greater than 0")

    @staticmethod
    def validate_equal_length_of_containers(first_container, second_container):
        if len(first_container) is 0:
            raise EmptyContainerException(type(first_container), "Container of type: {} should not be empty".format(
                    type(first_container)))
        if len(second_container) is 0:
            raise EmptyContainerException(type(second_container), "Container of type: {} should not be empty".format(
                    type(second_container)))
        if len(first_container) is not len(second_container):
            raise ArgumentsNotEqualException(
                type(first_container), "Lengths of the containers of type: {} should be equal".format(
                    type(first_container))
            )

    @staticmethod
    def most_common(k, dist_vector):
        common_labels = dict()

        for i, dist in enumerate(dist_vector):
            common_labels.setdefault(dist.label, 0)
            common_labels[dist.label] += 1
            if i >= k - 1:
                break

        return sorted(common_labels.items(), key=o.itemgetter(1))[0][0]

    @staticmethod
    def count_score(predictions, labels):
        score = 0

        for i, p in enumerate(predictions):
            if p == labels[i]:
                score += 1
        return score/len(predictions)

    def change_k(self, k):
        self.k = k

    def change_metrics(self, metrics):
        self.metrics = metrics

    def predict(self, new_data):
        new_data = new_data.astype(float)
        self.std_sc.fit(new_data)
        new_data = self.std_sc.transform(new_data)
        result = []

        for unknown in new_data:
            y = []
            for i, sample in enumerate(self.data):
                y.append(Sample(self.metrics(unknown, sample), self.target[i]))
            y.sort(key=o.attrgetter("dist"))
            prediction = KNN.most_common(self.k, y)
            result.append(prediction)

        return result

    def score(self, test_data, labels):
        result = self.predict(test_data)

        try:
            KNN.validate_equal_length_of_containers(test_data, labels)
        except EmptyContainerException as e:
            print(e.message)
            os._exit(1)
        except ArgumentsNotEqualException as e:
            print(e.message)
            os._exit(1)

        return KNN.count_score(result, labels)