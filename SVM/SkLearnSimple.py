'''
Created on 2019年4月5日

@author: Administrator
'''
from sklearn import svm
import numpy as np

x = [[2, 0], [1, 1], [2, 3]]
y = [0, 0, 1]
clf = svm.SVC(kernel='linear')
clf.fit(x, y)
print(clf)
#get support vectors
print(clf.support_vectors_)
#get indices of support vectors
print(clf.support_)
#get number of support vectors for each class
print(clf.n_support_)

predictData = np.array([0.5, 0.5]).reshape(1, -1)
predict = clf.predict(predictData)
print(predict)

