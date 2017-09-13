#!/usr/bin/python3.5
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import RidgeCV


class Logistic:
    def __init__(self):
        data = np.loadtxt("logisticReg_linear.txt", delimiter=',')
        self.X = data[:, :-1]
        self.Y = data[:, -1]
        self.X_copy = self.X
        self.X = np.column_stack((np.ones((self.X.shape[0], 1)), self.X))
        self.theta = np.zeros((self.X.shape[1], 1))
        self.alpha = 0.0011
        self.m = self.X.shape[0]
        self.J=[]

    def plotData(self):
        X1_1 = self.X[self.Y == 1, 2]
        X1_0 = self.X[self.Y == 0, 2]
        X0_1 = self.X[self.Y == 1, 1]
        X0_0 = self.X[self.Y == 0, 1]
        plt.plot(X0_1, X1_1, 'rx')
        plt.plot(X0_0, X1_0, 'o')
        plt.axis([30, 100, 20, 100])
        self.Y.shape = (self.Y.shape[0],1)
        plt.show()


    def hypothesis(self):
        return 1/(1+(np.exp(-np.dot(self.X, self.theta))))

    def gradient(self):
        return np.dot(self.X.T, (self.hypothesis() - self.Y))

    def costFunction(self):
        h = self.hypothesis().reshape(100, )
        # print(h.shape, self.Y.T.shape)
        J = (-1/self.m)*(np.dot(self.Y.T, np.log(h)) + np.dot((1-self.Y).T, np.log(1 - h)))
        # print(J)
        self.J.append(J)

    def batchGradientDescent(self,iter1):
        self.iter = iter1
        for i in range(iter1):
            self.theta = self.theta - (self.alpha/self.m) * self.gradient()
            self.costFunction()
        print("a=",self.theta)

    def plotCurve(self):
        self.Y.shape = (self.Y.shape[0],)
        # plt.axis([30, 100, 20, 100])
        X1_1 = self.X[self.Y == 1, 2]
        X1_0 = self.X[self.Y == 0, 2]
        X0_1 = self.X[self.Y == 1, 1]
        X0_0 = self.X[self.Y == 0, 1]

        plt.plot(X0_1, X1_1, 'rx')
        plt.plot(X0_0, X1_0, 'o')
        plt.title("batch gradient prediction")
        xAxis = np.arange(np.amin(self.X[:, 1]), np.amax(self.X[:, 1]), 0.01)
        yAxis = (-1/self.theta[2])*(self.theta[1] * xAxis + self.theta[0])
        plt.plot(xAxis, yAxis)
        plt.show()

    def plotCost(self):
        xAxis = np.arange(0,self.iter)
        # print(self.J)
        plt.title("Cost Function vs iter")
        plt.plot(list(xAxis), list(self.J))
        plt.show()

class skikitLogistic:
    def __init__(self):
        data = np.loadtxt("logisticReg_linear.txt", delimiter=',')
        self.X = data[:, :-1]
        self.Y = data[:, -1]
        self.X_copy = self.X
        self.m = self.X.shape[0]

    def findFit(self):
        regr = linear_model.LogisticRegression(solver='lbfgs')
        regr.fit(self.X, self.Y)
        self.predicted_y = regr.predict(self.X)
        self.theta1 = np.column_stack((regr.intercept_, regr.coef_)).T
        self.accuracy = regr.score(self.X, self.Y)
        print(self.accuracy)

    def plotCurve(self):
        self.Y.shape = (self.Y.shape[0],)
        X1_1 = self.X[self.Y == 1, 1]
        X1_0 = self.X[self.Y == 0, 1]
        X0_1 = self.X[self.Y == 1, 0]
        X0_0 = self.X[self.Y == 0, 0]

        plt.plot(X0_1, X1_1, 'rx')
        plt.plot(X0_0, X1_0, 'o')
        plt.title("skikit prediction")
        xAxis = np.arange(np.amin(self.X[:, 0]), np.amax(self.X[:, 0]), 0.01)
        print("theta1=", self.theta1)
        yAxis = (-1 / self.theta1[2]) * (self.theta1[1] * xAxis + self.theta1[0])
        plt.plot(xAxis, yAxis)
        plt.show()


def callLogistic():
    data_logistic = Logistic()
    data_logistic.plotData()
    data_logistic.batchGradientDescent(2000000)
    data_logistic.plotCurve()
    data_logistic.plotCost()

def callSkikitLogistic():
    data_logistic = skikitLogistic()
    data_logistic.findFit()
    data_logistic.plotCurve()

# callLogistic()
# callSkikitLogistic()



