from sklearn import svm
from process import create_data
import numpy as np

x_train, y_train, x_test, y_test = create_data()
print(y_train)
x_train = [a.A for a in x_train]
x_train = [a.reshape(-1) for a in x_train]
x_test = [a.A for a in x_test]
x_test = [a.reshape(-1) for a in x_test]

clf = svm.SVC(verbose=True)
clf.fit(x_train, y_train)

print(np.shape(x_test))
print(np.shape(x_test[0]))

preds = clf.predict(x_test)
print(preds)
