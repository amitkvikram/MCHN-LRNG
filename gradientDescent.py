#!/usr/bin/python3.5
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt



data= np.loadtxt('linearReg_m.txt',delimiter=',');
X= data[:,0];
Y= data[:,1];
X1=X;
plt.plot(X,Y,'rx');

m= np.shape(X)[0];

theta = np.zeros((2,1));
X= np.reshape(X,(len(X),1));
Y= np.reshape(Y,(len(Y),1));
X= np.column_stack((np.ones((len(X),1)),X));


plt.show();
alpha= 0.01;
J = [];
iter = np.arange(0,0,1);

def cost(theta1):
    #m= np.shape(X)[0];
    Cost= (1/(2*m))*(np.sum(np.square(np.dot(X,theta1)-Y)));
    print(Cost);
    J.append(Cost);

def gradientDescent1(X, Y, theta, m, alpha):
    i=0;
    for i in range(0):
        theta = theta - (alpha/m)*(np.dot(np.transpose(X),(np.dot(X,theta)-Y)));
        cost(theta);
    return theta;

theta= gradientDescent1(X,Y, theta, m, alpha);
plt.plot(iter, J);
plt.show();

plt.plot(X1,Y,'rx');
Xaxis= np.arange(0,np.amax(X),0.01);
Xaxis= np.reshape(Xaxis, (Xaxis.shape[0],1));
Xaxis1= np.column_stack((np.ones((len(Xaxis),1)),Xaxis));
h_theta= np.dot(Xaxis1,theta);

print(Xaxis.shape);
print(h_theta.shape);
plt.plot(Xaxis, h_theta);
plt.show();
