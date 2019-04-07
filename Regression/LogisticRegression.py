import numpy as np
import random

# m denotes the number fo examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    print('xTrans: ', xTrans)
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y        
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # but to be consistent with the gradient, i include it)
        cost = np.sum(loss ** 2) / (2 * m)
        if i % 10000 == 1:
            #print('hypothesis: ', hypothesis)
            #print('gradient: ', gradient)
            print("iteration %d / cost: %f" %(i, cost))
    return theta

def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 3))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias features
        x[i][0] = 1
        x[i][1] = i
        x[i][2] = numPoints - i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

def predict(x, theta):
    y = 0
    for i in range(0, len(theta)):
        y += x[i]*theta[i]
    return y

# generate 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = genData(100, 25, 10)
print("x: ", x)
print("y: ", y)
m, n = np.shape(x)
n_y = np.shape(y)
numIterations = 100000
alpha = 0.0001
theta = np.ones(n)
print("m:", m, " n:", n, " n_y:", n_y, " numIterations:", numIterations, " alpha:", alpha, " theta:", theta)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print('theta: ', theta)
yy = []
y_hat = []
for i in range(0, len(y)):
    yy.append(y[i])
    y_hat.append(predict(x[i], theta))
print("y:     ", yy)
print("y_hat: ", y_hat)