print(__doc__)

import numpy as np
import pylab as pl
from sklearn import svm

# create 40 separable points
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

print("X:", X)
print("Y:", Y)

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
# clf.intercept_[0] is w[2] or b
yy = a * xx - (clf.intercept_[0]) / w[1]
#print("yy:", yy)
# plot the parallels to the separating hyperplane that pass through the 
# support vectors
b = clf.support_vectors_[0]
print('support_vectors_', clf.support_vectors_)
yy_down = a * xx + (b[1] - a * b[0])
print('b', b)
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
pl.plot(xx, yy, 'k--')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=80, facecolors='red')
pl.scatter(X[:,0], X[:,1], c=Y, cmap=pl.cm.Paired)
pl.axis('tight')
pl.show()