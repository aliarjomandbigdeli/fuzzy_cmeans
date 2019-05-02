import random

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 5/1/2019
"""


class FCM:
    def __init__(self):
        self._u = []  # membership matrix
        self._u_previous = []
        self._cluster_centers = []
        self._fuzziness = 2
        self._error = 0.01
        self._cluster_number = 3
        self._dimension = 2  # number of features
        self.data = []

    def exec(self, cluster_number, max_iter, data):
        self._cluster_number = cluster_number
        self.data = data

        self.initialize_membership()

        for i in range(max_iter):
            self.calculate_cluster_centers()

            self.update_membership_values()

            if self.check_convergence() < self._error:
                break

    def create_random_data(self, num_of_data, dimenstion, cluster_number):
        self._dimension = dimenstion

    def initialize_membership(self):
        # for i in range(len(self.data)):
        for i in range(4):
            sum = 0
            self._u.append([])
            for j in range(self._cluster_number):
                self._u[i].append(random.uniform(1.0, 2.0))
                sum += self._u[i][j]
            for j in range(self._cluster_number):
                self._u[i][j] /= sum

        # for k in range(4):
        #     print('\n')
        #     for l in range(self._cluster_number):
        #         print(self._u[k][l])
        print('initialize_membership')

    def calculate_cluster_centers(self):
        self._cluster_centers.clear()
        for i in range(self._cluster_number):
            cluster_center_row = []
            for j in range(self._dimension):
                sum1 = 0
                sum2 = 0
                for k in range(len(self.data)):
                    powered = self._u[k][i] ** self._fuzziness
                    sum1 += powered * self.data[k][i]
                    sum2 += powered
                cluster_center_row.append(sum1 / sum2)
            self._cluster_centers.append(cluster_center_row)

        print('calculate_cluster_centers')

    def update_membership_values(self):
        for i in range(len(self.data)):
            for j in range(self._cluster_number):
                self._u_previous[i][j] = self._u[i][j]
                sum = 0
                upper = self.distance(self.data[i], self._cluster_centers[j])
                for k in range(self._cluster_number):
                    lower = self.distance(self.data[i], self._cluster_centers[k])
                    sum += (upper / lower) ** (2 / (self._fuzziness - 1))
                self._u[i][j] = 1 / sum

        print('update_membership_values')

    def distance(self, p1, p2):
        sum = 0
        for i in range(len(p1)):
            sum += (p1[i] - p2[i]) ** 2
        return sum ** 0.5

    def check_convergence(self):
        print('check_convergence')
        sum = 0
        for i in range(len(self.data)):
            for j in range(self._cluster_number):
                sum += (self._u[i][j] - self._u_previous[i][j]) ** 2
        return sum


def main():
    # my_fcm = FCM()
    # my_fcm.initialize_membership()
    print('hello')


if __name__ == '__main__':
    main()
