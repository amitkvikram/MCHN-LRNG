#!/usr/bin/python3.5
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import RidgeCV

class uniVariate():
    def __init__(self):
        self.data= np.loadtxt("nonlinearReg_uni1.txt")
        self.X= self.data[1:,-1]
        self.Y= self.data[1:,-1:]
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
        print("Theta=",self.theta)
    def plotCurve(self):
        plt.plot(self.X1, self.Y, 'rx')
        xAxis= xAxis1=np.arange(0,np.amax(self.X1),0.01)
        xAxis.shape=(xAxis.shape[0],1)
        xAxis= self.mapfeature(2,xAxis)
        xAxis[:,1:]= xAxis[:,1:]/self.sigma       #Feature Scaling
        yAxis= np.dot(xAxis, self.theta)
        # print(xAxis[:,1:2].ravel().shape, yAxis.shape)
        plt.plot(xAxis1 , list(yAxis))
        plt.show()

class uni_skikit():
    def __init__(self):
        self.data= np.loadtxt("nonlinearReg_uni1.txt")
        self.X= self.data[1:,:-1]
        self.Y= self.data[1:,-1:]
        self.X1= self.X;
    def mapfeature(self):
        poly = PolynomialFeatures(3)
        self.X_new = poly.fit_transform(self.X)
        print(self.X_new.shape)
    def findFit(self):
        regr= linear_model.RidgeCV(alphas=[100,10],fit_intercept=True);
        regr.fit(self.X_new, self.Y);
        # print(regr.alpha_)
        self.accuracy= regr.score(self.X_new[-20:,:], self.Y[-20:,:])
        print("skikit optTheta=",self.accuracy)
        return regr;
    def plotCurve(self):
        poly = PolynomialFeatures(3)
        plt.plot(self.X1, self.Y, 'rx')
        xAxis= xAxis1=np.arange(0,np.amax(self.X1),0.01).reshape(-1, 1)

        xAxis= poly.fit_transform(xAxis)
        yAxis = self.regr.predict(xAxis)
        plt.plot(xAxis1 , list(yAxis))
        plt.show()

def uni_grad():
    data_uni = uniVariate()
    data_uni.alpha(0.01)
    data_uni.plotData()
    data_uni.X=data_uni.mapfeature(2, data_uni.X)
    # print(data_uni.X)
    data_uni.featureNormalize();
    data_uni.gradientDescent(200000);
    data_uni.plotCurve()

def skikit_nonLinearReg():
    data_uni1= uni_skikit();
    data_uni1.mapfeature();
    data_uni1.regr= data_uni1.findFit();
    data_uni1.plotCurve();

# skikit_nonLinearReg()
# uni_grad();

##################################################################################################
class multi_skikit():
    def __init__(self):
        self.data= np.loadtxt("nonlinearReg_poly.txt")
        self.X= self.data[1:,:-1]
        self.Y= self.data[1:,-1:]

    def mapFeature(self):
        poly= PolynomialFeatures(3)
        self.X_new= poly.fit_transform(self.X)

    def findFit(self):
        regr= linear_model.Ridge(alpha=10000.0)
        regr.fit(self.X_new, self.Y)
        self.accuracy = regr.score(self.X_new, self.Y)
        print(self.accuracy)

def skikit_nonLinearReg1():
    data_multi1= multi_skikit();
    data_multi1.mapFeature();
    data_multi1.regr = data_multi1.findFit();

skikit_nonLinearReg1();