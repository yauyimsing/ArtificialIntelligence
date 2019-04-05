'''
Created on 2019年4月4日

@author: Administrator
'''
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()

print(iris['target_names'])

knn.fit(iris.data, iris.target)

predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])

print("predictedLabel: " , predictedLabel)
