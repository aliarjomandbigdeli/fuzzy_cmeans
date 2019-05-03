from sklearn.datasets.samples_generator import make_blobs
from numpy import linalg as la
import random

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 5/1/2019
"""


def distance(p1, p2):
    """
    this function calculate distance based on 2-norm of vectors definition
    :parameter p1: is a list
    :parameter p2: is a list
    """
    s = 0
    for i in range(len(p1)):
        s += (p1[i] - p2[i]) ** 2
    return s ** 0.5


class FCM:
    """
    This is a class for fuzzy c-means clustering.
    This code implemented based on https://home.deib.polimi.it/matteucc/Clustering/tutorial_html/cmeans.html

    Attributes:
        _u: membership matrix in the format of list of list.
        _data: your dataset in the format of list of list for 2D input, for other dimension you can input similar
         format. You set it or use create_random_data method.
        _dimension: dimension of data or number of features of data
        _fuzzifier (m): is the hyper- parameter that controls how fuzzy the cluster will be. The higher it is,
         the fuzzier the cluster will be in the end.
    """

    def __init__(self):
        self._u = []  # membership matrix
        self._u_previous = []  # membership matrix in previous iteration
        self._data = []
        self._cluster_centers = []
        self._fuzzifier = 2
        self._error = 0.01
        self._cluster_number = 0
        self._dimension = 2  # number of features

    def u(self):
        """:returns membership matrix"""
        return self._u

    def data(self, d=None):
        """getter and setter of data"""
        if d:
            self._data = d
        return self._data

    def cluster_centers(self):
        """:returns cluster centers"""
        return self._cluster_centers

    def exec(self, cluster_number, max_iter, data):
        self._cluster_number = cluster_number
        self._data = data

        self.initialize_membership()

        for i in range(max_iter):
            self.calculate_cluster_centers()

            self.update_membership_values()

            if self.check_convergence() < self._error:
                break

    def create_random_data(self, num_of_data, dimension, cluster_number):
        x, y = make_blobs(n_samples=num_of_data, centers=cluster_number, n_features=dimension)
        self._dimension = dimension
        self._data = x

    def initialize_membership(self):
        max_range = max(self._data[:, 0])
        min_range = min(self._data[:, 0])

        for i in range(len(self._data)):
            s = 0
            self._u.append([])
            self._u_previous.append([])
            for j in range(self._cluster_number):
                self._u[i].append(random.uniform(min_range, max_range))
                s += self._u[i][j]
                self._u_previous[i].append(0)
            for j in range(self._cluster_number):
                self._u[i][j] /= s

    def calculate_cluster_centers(self):
        self._cluster_centers.clear()
        for i in range(self._cluster_number):
            cluster_center_row = []
            for j in range(self._dimension):
                sum1 = 0
                sum2 = 0
                for k in range(len(self._data)):
                    powered = self._u[k][i] ** self._fuzzifier
                    sum1 += powered * self._data[k][j]
                    sum2 += powered
                cluster_center_row.append(sum1 / sum2)
            self._cluster_centers.append(cluster_center_row)

    def update_membership_values(self):
        for i in range(len(self._data)):
            for j in range(self._cluster_number):
                self._u_previous[i][j] = self._u[i][j]
                s = 0
                upper = distance(self._data[i], self._cluster_centers[j])
                for k in range(self._cluster_number):
                    lower = distance(self._data[i], self._cluster_centers[k])
                    s += (upper / lower) ** (2 / (self._fuzzifier - 1))
                self._u[i][j] = 1 / s

    def check_convergence(self):
        return abs(la.norm(self._u, ord=2) - la.norm(self._u_previous, ord=2))
