import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from NeuralNetwork import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import classification_report

digits = load_digits()
X = digits.data
y = digits.target
print("X: " , X.shape)
print('y: ', y.shape)
X -= X.min() # normalize the values to bring three into the range 0-1

X /= X.max()
#print("X.min: ", X)

nn = NeuralNetwork([64,100,10], 'logistic')
X_train, X_test, y_train, y_test = train_test_split(X, y)
print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)
#print('y_test: ', y_test)
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
#print('labels_test: ', labels_test.shape)
print('start fitting')
nn.fit(X_train, labels_train, epochs=3000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))
    #print('o: ', o)
    #print('argmax(o): ', np.argmax(o))
#print('y_test: ', y_test)
print('predictions: ', predictions)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

      

