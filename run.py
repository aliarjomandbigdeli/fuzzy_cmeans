import matplotlib.pyplot as plt
import FCMeans

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 5/1/2019
"""


def main():
    cluster_num = input("Enter number of cluster: ")
    data_num = input("Enter number of data(data set size): ")

    my_fcm = FCMeans.FCM()
    my_fcm.create_random_data(int(data_num), 2, int(cluster_num))
    my_fcm.exec(int(cluster_num), 100, my_fcm.data())

    for pt in my_fcm.data():
        plt.plot(pt[0], pt[1], 'bo')
    for ct in my_fcm.cluster_centers():
        plt.plot(ct[0], ct[1], 'r*')
    plt.grid(b=None, which='both', axis='both', color='gray', linestyle='-', linewidth=2)
    plt.title('number of clusters: ' + cluster_num + ', number of data: ' + data_num)
    plt.show()


if __name__ == '__main__':
    main()
