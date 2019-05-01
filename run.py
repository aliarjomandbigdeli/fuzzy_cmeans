"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 5/1/2019
"""


class FCM:
    def __init__(self):
        self._fuzziness = 2
        self._error = 0.01
        self._cluster_number = None
        self._data = None

    def exec(self, cluster_number, max_iter, data):
        self._cluster_number = cluster_number
        self._data = data

        self.initialize_membership()

        for i in range(max_iter):
            self.calculate_cluster_centers()

            self.update_membership_values()

            if self.check_convergence() < self._error:
                break

    def initialize_membership(self):
        print('initialize_membership')

    def calculate_cluster_centers(self):
        print('calculate_cluster_centers')

    def update_membership_values(self):
        print('update_membership_values')

    def check_convergence(self):
        print('check_convergence')
        return 0


def main():
    print('hello')


if __name__ == '__main__':
    main()
