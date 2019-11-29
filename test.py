import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
def plot_data(x, y):

    fig = plt.figure()
    plt.scatter(x, y)
    plt.xlabel("Profit in $10,000")
    plt.ylabel("Population of City in 10,000s")
    plt.show()


print(plot_data(X, Y))
m = Y.size
X = np.stack([np.ones(m), X], axis=1)
print(X[0][1])
print(Y[0])
def cost_function(x, y, weight, bias):
    theta = np.array([weight, bias])
    k = y.size
    j = 0
    j = (1/(2 * k) * np.sum(np.square(np.dot(x, theta) - y)))
    return j


print(cost_function(X, Y, 0, 0))
print(cost_function(X, Y, 20, 20))
# Matrix
def minimize(x, y, weight, bias, learn_rate, iteration):
    theta = np.array([weight, bias])
    data_size = y.shape[0]
    print(data_size)
    j_history = []
    theta_history = []
    for i in range(iteration):
        theta = theta - (learn_rate / data_size) * (np.dot(x, theta) - y).dot(X)
        j_history.append(cost_function(x, y, theta[0], theta[1]))
        theta_history.append(theta)
    return theta


print(minimize(X, Y, 0, 0, 0.01, 1500))
# equation
def test(x, y, weight, bias, learn_rate, iteration):
    size = y.shape[0]
    for i in range(iteration):
        sum_weight = sum((((weight * x[j][1] + bias) - y[j]) * x[j][1]) for j in range(size))
        sum_bias = sum(((weight * x[j][1] + bias) - y[j]) for j in range(size))
        weight = weight - (learn_rate/size) * sum_weight
        bias = bias - (learn_rate/size) * sum_bias
    theta = np.array([bias, weight])
    return theta


print(test(X, Y, 0, 0, 0.01, 1500))
predict1 = np.dot([1, 3.5], test(X, Y, 0, 0, 0.01, 1500))
print(predict1)
choose = input("input some population : ")
predict2 = (test(X, Y, 0, 0, 0.01, 1500)[1] * float(choose)) + test(X, Y, 0, 0, 0.01, 1500)[0]
print(predict2)


# plot the linear fit
plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], np.dot(X, test(X, Y, 0, 0, 0.01, 1500)), '-')
plt.legend(['Linear regression', 'Training data'])
plt.show()
