import numpy as np
import pandas as pd
data = pd.read_csv('exer2.txt', sep = ',', header = None)
X = data.iloc[:,0:2] # read first two columns into X
Y = data.iloc[:,2] # read the third column into y
m = len(Y) # no. of training samples
X = (X - np.mean(X))/np.std(X)
X.insert(0, "X0", 1, True)
np.array(X)
print(X.shape)
theta = np.array([0,0,0])
print(theta.shape)
np.array(Y)
print(Y.shape)
print(np.dot(X, theta).shape)
def computeCostMulti(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)
J = computeCostMulti(X, Y, theta)
print(J)
def gradientDescentMulti(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(temp, X)
        theta = theta - (alpha/m) * temp
    return theta
theta = gradientDescentMulti(X, Y, theta, 0.01, 14000)
print(theta)
J = computeCostMulti(X, Y, theta)
print(J)
print(np.dot(X[0:1],theta))
X[0:1]
