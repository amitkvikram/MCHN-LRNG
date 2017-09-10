#!/usr/bin/python3.5
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

class uniVariate():
    def __init__(self):
        self.data= np.loadtxt("nonlinearReg_uni.txt")
        self.X= self.data[:,:-1]
        self.Y= self.data[:,-1:]
        self.X1= self.X;
    def alpha(self, alpha):
        self.alpha = alpha
        self.J = []
    def plotData(self):
        plt.plot(self.X, self.Y, 'rx')
        plt.show()
    def mapfeature(self, pow, X):
        self.pow=pow
        for i in range(pow-1):
            X=np.column_stack((X, np.power(X[:,-1:],i+2)))
        X = np.column_stack((np.ones((X.shape[0], 1)), X));   #adding X0
        return X;

    def featureNormalize(self):
        max1= np.amax(self.X[:,1:], 0)
        min1= np.amin(self.X[:,1:],0)
        print(max1, min1)
        self.sigma = max1 - min1
        self.X[:,1:]= self.X[:,1:]/self.sigma;
    def costCalc(self):
        J= (1/(2*self.m))*np.sum(np.square(np.dot(self.X, self.theta) - self.Y))
        self.J.append(J)
    def gradientDescent(self, iter1):
        self.theta= np.zeros((1+self.pow,1))
        self.iter=iter1;
        m=self.m = self.X.shape[0]
        for i in range(iter1):
            self.theta=self.theta-(self.alpha/m)*(np.dot(self.X.T, (np.dot(self.X, self.theta) - self.Y)));
            self.costCalc()
    def plotCurve(self):
        plt.plot(self.X1, self.Y, 'rx')
        xAxis= xAxis1=np.arange(0,np.amax(self.X1),0.01)
        print(np.amax(self.X1))
        xAxis.shape=(xAxis.shape[0],1)
        xAxis= self.mapfeature(3,xAxis)
        xAxis[:,1:]= xAxis[:,1:]/self.sigma       #Feature Scaling
        print(xAxis.shape)
        yAxis= np.dot(xAxis, self.theta)
        print(xAxis[:,1:2].ravel().shape, yAxis.shape)
        plt.plot(xAxis1 , list(yAxis))
        plt.show()



data_uni = uniVariate()
data_uni.alpha(0.01)
data_uni.plotData()
data_uni.X=data_uni.mapfeature(3, data_uni.X)
# print(data_uni.X)
data_uni.featureNormalize();
# print(data_uni.X)
data_uni.gradientDescent(200000);
data_uni.plotCurve()
