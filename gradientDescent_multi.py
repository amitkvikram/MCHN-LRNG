#!/usr/bin/python3.5
import numpy as np
import matplotlib as plt
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import style
plt.style.use('classic')

data = np.loadtxt('ex1data2.txt', delimiter=',');
Y= data[:,-1];
X= data[:,:-1];

theta_0= np.zeros((500,1));
theta_1= np.zeros((500,1));

Y= np.reshape(Y,(len(Y),1));
theta= np.zeros((X.shape[1],1));
X= np.column_stack((np.ones((len(X),1)),X));
m= np.shape(X)[0];
alpha= 0.008;
J=[];

def featureScaling(X):
    max= np.amax(X, axis=0);
    min= np.amin(X, axis=0);
    S_dev= max-min;
    return (X/S_dev);


X= featureScaling(X);
# print(X);


def cost(theta1):
    #m= np.shape(X)[0];
    return (1/(2*m))*(np.sum(np.square(np.dot(X,theta1)-Y)));J.append(Cost);


# Theta0= np.arange(0,10,1);
# Theta0.shape=(10, 1);
# print(Theta0.shape);
# Theta1= np.arange(0,10,0.01);
# # J1= np.zeros((len(Theta0),len(Theta1)));
# J1=[];
# for i in Theta0:
#     for j in Theta1:
#         J1.append(cost(np.array([i, j])));
#
# fig= plt.figure(1)
# ax= fig.gca(projection="3d",title="3D1",xlabel="x",axisbg='gray')
# ax.plot_surface(Theta0,Theta1,J1,color="red")
# plt.show();


def gradientDescent1(X, Y, theta, m, alpha):
    i=0;
    for i in range(500):

        theta = theta - (alpha/m)*(np.dot(np.transpose(X),(np.dot(X,theta)-Y)));
        theta_0[i,0]= theta[0,:];
        theta_1[i,0]= theta[1,:];
        J.append(cost(theta));
    return theta;

theta= gradientDescent1(X,Y, theta, m ,alpha);
iter=np.arange(0,500,1);
plt.plot(iter, J);
plt.show();


fig= plt.figure(2)
ax= fig.gca(projection="3d",title="3D1",xlabel="x",axisbg='gray')
line,=plt.plot(theta_0,theta_1,J,'^',color='r')
plt.show();
