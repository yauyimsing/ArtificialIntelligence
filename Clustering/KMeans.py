import numpy as np
from conda.connection import dist

# Function: K Means
# ---------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the 
# dataset which are similar to one another).

# define kmeans method (dataset, devide k class, stop condition)
def kmeans(X, k, maxIt):
    numPoints, numDim = X.shape # get rows and columns from instance
    dataSet = np.zeros((numPoints, numDim+1))
    dataSet[:, :-1] = X
    # initialize centroids randomly
    centroids = dataSet[np.random.randint(numPoints, size=k), :]
    #print(np.random.randint(numPoints, size=k))
    centroids = dataSet[0:2, :]
    # randomly assign labels to initial centroid
    centroids[:, -1] = range(1, k+1)
    #print('centroids: ', centroids)
    # initialize book keeping vars.
    iterations = 0
    oldCentroids = None
    
    # run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        print("iterations: ", iterations)
        print("dataSet: ", dataSet)
        print("centroids: ", centroids)
        
        # save old centroids for convergence test. book keeping
        oldCentroids = np.copy(centroids)
        iterations += 1
        # assign labels to each datapoint based on centroids
        updateLabels(dataSet, centroids)
        # assign centroids based on datapoint labels
        centroids = getCentroids(dataSet, k)
    return dataSet
    
# function: should stop
# ----------------
# returns true or false if k-means is done. k-means terminates either
# because it has run a maximum number of iterations or the centroids
# top changing.
def shouldStop(oldCentroids, centroids, iterations, maxIt):
    if iterations > maxIt:
        return True
    flag = np.array_equal(oldCentroids, centroids)
    #print('flag: ', flag)
    return flag

# function: get labels
# ---------------
# update a label for each piece of data in the dataset.
def updateLabels(dataSet, centroids):
    # for each element in the dataset, chose the closest centroid.
    # make that centroid the element's label
    numPoints, numDim = dataSet.shape
    for i in range(0, numPoints):
        #print("dataSet[i, :-1], centroids", dataSet[i, :], centroids)
        dataSet[i, -1] = getLabelFromClosestCentroid(dataSet[i, :-1], centroids)

def getLabelFromClosestCentroid(dataSetRow, centroids):
    label = centroids[0, -1]
    minDist = np.linalg.norm(dataSetRow - centroids[0, :-1])
    for i in range(1, centroids.shape[0]):
        dist = np.linalg.norm(dataSetRow - centroids[i, :-1])
        if dist < minDist:
            minDist = dist
            label = centroids[i, -1]
    print('minDist: ', minDist)
    return label
        
# function: get centroids
# ----------------
# returns k random centrois, each of dimension n.
def getCentroids(dataSet, k):
    # each centroid is the geometric mean of the points that
    # have that centroid's label. important: if a centroid is empty (no points have
    # that centroid's labels) you should randomly re-initialize it.
    result = np.zeros((k, dataSet.shape[1]))
    #print('result: ', result)
    for i in range(1, k + 1):
        oneCluster = dataSet[dataSet[:, -1] == i, :-1]
        #print('oneCluster: ', oneCluster)
        result[i - 1, :-1] = np.mean(oneCluster, axis=0)
        result[i-1, -1] = i
        #print('result: ', result)
    return result

def main():
    x1 = np.array([1, 1])
    x2 = np.array([2, 1])
    x3 = np.array([4, 3])
    x4 = np.array([5, 4])
    testX = np.vstack((x1, x2, x3, x4))
    #print("testX: ", testX)
    result = kmeans(testX, 2, 10)
    print('final result:')
    print(result)
    
if __name__ == "__main__":
    main()