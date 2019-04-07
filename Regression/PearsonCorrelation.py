import numpy as np
from astropy.units import Ybarn
import math

def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar**2
        varY += diffYYBar**2
    
    SST = math.sqrt(varX * varY)
    return SSR / SST

# polynomial regression
def polyfit(x, y, degree=1):
    """
    parma: degree represents the power of the expression
    """
    results = {}
    # get coefficients of the expression from x and y
    coeffs = np.polyfit(x, y, degree)
    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()
    # r-squared
    # get the expression p
    p = np.poly1d(coeffs)
    print('p: ', p)
    # fit values, and mean
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y-ybar)**2)
    results['determination'] = ssreg / sstot
    return results

testX = [1, 3, 8, 7, 9]
testY = [10, 12, 24, 21, 34]
print("r: ", computeCorrelation(testX, testY))
print("r^2: ", computeCorrelation(testX, testY)**2)
results = polyfit(testX, testY, 1)
print(results)
