from cat.lr_utils import load_dataset
import numpy as np
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
x_train = np.array([x.ravel()/255 for x in train_set_x_orig])
x_test = np.array([x.ravel()/255 for x in test_set_x_orig])

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def cost_fun(a,y):
    m = len(y)
    return -(1/m)*np.sum(y*np.log(a)+(1-y)*np.log(1-a))
def fp(x,w1,b1,w2,b2):
    z1 = x.dot(w1)+b1
    a1 = sigmoid(z1)
    z2 = a1.dot(w2)+b2
    a2 = sigmoid(z2)
    return a1,a2
def bp(x,a1,a2,w1,b1,w2,b2,y,alpha):
    m = len(y)
    delta2 = a2-y
    delta1 = delta2.dot(w2.T)*(a1*(1-a1))
    dw2 = (1/m)*a1.T.dot(delta2)
    db2 = (1/m)*np.sum(delta2)
    dw1 = (1/m)*x.T.dot(delta1)
    db1 = (1/m)*np.sum(delta1)
    w2 = w2 - alpha*dw2
    b2 = b2 - alpha*db2
    w1 = w1 - alpha*dw1
    b1 = b1 - alpha*db1
    return w1,b1,w2,b2
x = x_train
y = train_set_y_orig.T
m,n = x.shape
max_iter = 10000
np.random.seed(41)
w1 = np.random.random((n,30))
b1 = np.random.random(30)
w2 = np.random.random((30,1))
b2 = np.random.random(1)
jd_history= []
for i in range(max_iter):
    a1,a2 = fp(x,w1,b1,w2,b2)
    cost = cost_fun(a2,y)
    w1, b1, w2, b2 = bp(x,a1,a2,w1,b1,w2,b2,y,alpha=0.01)
    jd_history.append(cost)

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
_,y1 = fp(x_train,w1,b1,w2,b2)
_,y2 = fp(x_test,w1,b1,w2,b2)
y1 = np.where(y1>0.5,1,0)
y2 = np.where(y2>0.5,1,0)
print(accuracy_score(y1,train_set_y_orig.T))
print(accuracy_score(y2,test_set_y_orig.T))
plt.plot(jd_history)
plt.show()



