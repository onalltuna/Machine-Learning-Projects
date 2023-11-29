import math
import matplotlib.pyplot as plt
import numpy as np

# read data into memory
data_set_train = np.genfromtxt(
    "hw03_data_set_train.csv", delimiter=",", skip_header=1)
data_set_test = np.genfromtxt(
    "hw03_data_set_test.csv", delimiter=",", skip_header=1)

# get x and y values
x_train = data_set_train[:, 0]
y_train = data_set_train[:, 1]
x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 1.6
maximum_value = 5.1
x_interval = np.arange(start=minimum_value, stop=maximum_value, step=0.001)


def plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(x_train, y_train, "b.", markersize=10)
    plt.plot(x_test, y_test, "r.", markersize=10)
    plt.plot(x_interval, y_interval_hat, "k-")
    plt.xlim([1.55, 5.15])
    plt.xlabel("Time (sec)")
    plt.ylabel("Signal (millivolt)")
    plt.legend(["training", "test"])
    plt.show()
    return (fig)


# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def regressogram(x_query, x_train, y_train, left_borders, right_borders):
    # your implementation starts below
    y_hat = np.zeros(len(x_query))

    # left_borders -= 0.1
    # right_borders -= 0.1
    # print("left_borders", left_borders)
    # print("right_borders", right_borders)

    for b in range(len(left_borders)):
        in_bin_train = np.zeros((len(x_train)))
        in_bin_query = np.zeros((len(x_query)))

        for x in range(len(x_train)):
            if (left_borders[b] <= x_train[x]) & (x_train[x] <= right_borders[b]):
                in_bin_train[x] = 1

        for x in range(len(x_query)):
            if (left_borders[b] <= x_query[x]) & (x_query[x] <= right_borders[b]):
                in_bin_query[x] = 1

        averge_y = np.average(y_train[in_bin_train == 1])

        for x in range(len(x_query)):
            if in_bin_query[x] == 1:
                y_hat[x] = averge_y

    # your implementation ends above
    return (y_hat)


bin_width = 0.35
left_borders = np.arange(
    start=minimum_value, stop=maximum_value, step=bin_width)
right_borders = np.arange(start=minimum_value + bin_width,
                          stop=maximum_value + bin_width, step=bin_width)

y_interval_hat = regressogram(
    x_interval, x_train, y_train, left_borders, right_borders)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("regressogram.pdf", bbox_inches="tight")

y_test_hat = regressogram(x_test, x_train, y_train,
                          left_borders, right_borders)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Regressogram => RMSE is {} when h is {}".format(rmse, bin_width))


# STEP 4
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def running_mean_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below

    y_hat = np.zeros(len(x_query))
    in_bin_train = np.zeros((len(x_train)))

    for i, x_q in enumerate(x_query):
        in_bin_train = np.zeros((len(x_train)))
        for j, x_t in enumerate(x_train):
            if (((x_t - x_q) / bin_width) >= -0.5) and (((x_t - x_q) / bin_width) < 0.5):
                in_bin_train[j] = 1

        averge_y = np.average(y_train[in_bin_train == 1])

        y_hat[i] = averge_y

    # your implementation ends above
    return (y_hat)


bin_width = 0.35

y_interval_hat = running_mean_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("running_mean_smoother.pdf", bbox_inches="tight")

y_test_hat = running_mean_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Running Mean Smoother => RMSE is {} when h is {}".format(rmse, bin_width))


# STEP 5
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def kernel_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below
    y_hat = np.zeros(len(x_query))

    for i, x_q in enumerate(x_query):
        in_bin_train = np.zeros((len(x_train)))
        in_bin_train2 = np.zeros((len(x_train)))
        for j, x_t in enumerate(x_train):
            x = (1.0 / np.sqrt(2 * math.pi) *
                               np.exp(-0.5 * (x_t - x_q)**2 / bin_width**2)) * y_train[j]
            y = (1.0 / np.sqrt(2 * math.pi) *
                               np.exp(-0.5 * (x_t - x_q)**2 / bin_width**2))
            in_bin_train[j] = x 
            in_bin_train2[j] = y
        
        y_hat[i] = np.sum(in_bin_train) / np.sum(in_bin_train2)

    # print("in_bin_train:", in_bin_train)
    # print("y_train: ", len(y_train))

    # your implementation ends above
    return (y_hat)


bin_width = 0.35

y_interval_hat = kernel_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("kernel_smoother.pdf", bbox_inches="tight")

y_test_hat = kernel_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Kernel Smoother => RMSE is {} when h is {}".format(rmse, bin_width))
