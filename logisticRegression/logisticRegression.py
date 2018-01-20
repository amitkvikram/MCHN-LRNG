#!/usr/bin/python3.5
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import RidgeCV
import matplotlib.cm as cm
import matplotlib.mlab as mlab


class Logistic:
    def __init__(self):
        data = np.loadtxt("logisticReg_linear.txt", delimiter=',')
        self.X = data[:, :-1]
        self.Y = data[:, -1]
        self.X_copy = self.X
        self.X = np.column_stack((np.ones((self.X.shape[0], 1)), self.X))
        self.theta = np.zeros((self.X.shape[1], 1))
        self.alpha = 0.00121
        self.m = self.X.shape[0]
        self.J=[]
        self.lambda1 = 100   #lambda1 is regularization parameter

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


class LogisticPoly:
    def __init__(self):
        self.data = np.loadtxt("logisticReg_poly.txt", delimiter=',')
        self.X = self.data[:, :-1]
        self.Y = self.data[:, -1]
        # self.plot_data()
        self.X = self.map_feature(6, self.X)
        self.featurScaling()
        self.X = np.column_stack((np.ones((self.X.shape[0], 1)), self.X))
        self.theta = np.zeros((self.X.shape[1], ))
        self.alpha = 0.00121
        self.lambda1 = .5                  #lambda1 is regularization parameter
        self.m = self.data.shape[0]
        self.J = []

    def plot_data(self):
        pos_X = self.X[self.Y==1, 0]
        pos_Y = self.X[self.Y==1, 1]
        neg_X = self.X[self.Y==0, 0]
        neg_Y = self.X[self.Y==0, 1]
        plt.plot(pos_X, pos_Y, 'bo')
        plt.plot(neg_X, neg_Y, 'rx')
        plt.show()

    def map_feature(self, degree, X):
        for i in range(2, degree + 1):
            for j in range(0, i+1):
                X = np.column_stack((X, np.multiply(np.power(X[:, 0], i - j), np.power(X[:, 1], j))))
        return X;

    def featurScaling(self):
        max1 = np.amax(self.X, 0)
        min1 = np.amin(self.X, 0)
        self.sigma = max1 - min1
        self.X = self.X / self.sigma

    def hypothesis(self, X):
        return 1/(1 + np.exp(-np.dot(X, self.theta)))

    def gradient(self):
        reg = self.lambda1*self.theta
        reg[0] = 0
        grad = np.dot(self.X.T, (self.hypothesis(self.X) - self.Y))  + reg
        return grad

    def gradient_descent(self, iter1):
        self.iter = iter1
        for i in range(iter1):
            self.theta = self.theta - (self.alpha/self.m) * self.gradient()
            self.cost()
        self.theta[1:,] = self.theta[1:,]/self.sigma
    def cost(self):
        h = self.hypothesis(self.X)
        J = (-1 / self.m) * (np.dot(self.Y.T, np.log(h)) + np.dot((1 - self.Y).T, np.log(1 - h)))
        self.J.append(J)

    def plot_curve(self):
        plt.figure()
        self.X[:,1:] = self.X[:, 1:]*self.sigma
        pos_X = self.X[self.Y == 1, 1]
        pos_Y = self.X[self.Y == 1, 2]
        neg_X = self.X[self.Y == 0, 1]
        neg_Y = self.X[self.Y == 0, 2]
        plt.plot(pos_X, pos_Y, 'rx')
        plt.plot(neg_X, neg_Y, 'bo')
        xAxis = np.linspace(np.min(self.data[:, 0]), np.max(self.data[:, 0]), 50)
        yAxis = np.linspace(np.min(self.data[:, 1]), np.max(self.data[:, 1]), 50)
        xAxis1 = xAxis
        yAxis1 = yAxis
        plt.title("decision Boundary_with Regularization")
        xAxis, yAxis = np.meshgrid(xAxis, xAxis)
        zAxis = np.zeros((50,50))
        for i in range(xAxis.shape[1]):
            temp_x = self.map_feature(6, np.column_stack((xAxis[:, i], yAxis[:, i])))
            temp_x = np.column_stack((np.ones((xAxis.shape[0], )), temp_x))
            zAxis[:, i] = np.dot(temp_x, self.theta)

        CS = plt.contour(xAxis1, yAxis1, zAxis, [0.0])
        # plt.colorbar(CS)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.show()

class SkikitPoly():
    def __init__(self,fname):
        self.data = np.loadtxt(fname, delimiter=',')
        self.X = self.data[:, :-1]
        self.Y = self.data[:, -1]

    def findFit(self):
        self.regr = linear_model.LogisticRegression(solver='newton-cg')
        self.regr.fit(self.X, self.Y)
        self.theta = self.regr.coef_
        intercept = self.regr.intercept_
        self.theta = np.column_stack((intercept, self.theta))
        self.theta.shape = (self.theta.shape[1], )
        print(self.theta.shape)
        self.score = self.regr.score(self.X, self.Y)
        print(self.score)

    def map_feature(self, deg):
        self.poly = PolynomialFeatures(deg, include_bias=False)
        self.X = self.poly.fit_transform(self.X)

    def plot_curve(self):
        pos_X = self.X[self.Y == 1, 0]
        pos_Y = self.X[self.Y == 1, 1]
        neg_X = self.X[self.Y == 0, 0]
        neg_Y = self.X[self.Y == 0, 1]
        plt.plot(pos_X, pos_Y, 'rx')
        plt.plot(neg_X, neg_Y, 'bo')
        xAxis = np.linspace(np.min(self.data[:, 0]), np.max(self.data[:, 0]), 50)
        # yAxis = np.linspace(np.min(self.data[:, 1]), np.max(self.data[:, 1]), 50)
        plt.title("Decision Boundary@sklearn")
        xAxis, yAxis = np.meshgrid(xAxis, xAxis)
        zAxis = np.zeros((50, 50))
        for i in range(xAxis.shape[1]):
            temp_x = self.poly.fit_transform(np.column_stack((xAxis[:, i], yAxis[:, i])))
            temp_x = np.column_stack((np.ones((xAxis.shape[0],)), temp_x))
            zAxis[:, i] = np.dot(temp_x, self.theta)

        CS = plt.contour(xAxis, yAxis, zAxis, [0.0])
        plt.clabel(CS, inline=1, fontsize=10)
        plt.show()

def callLogistic_poly():
    data_logistic = LogisticPoly()
    data_logistic.gradient_descent(3000)
    data_logistic.plot_curve()

def callskikit_poly():
    data_logistic = SkikitPoly("logisticReg_poly.txt")
    data_logistic.map_feature(6)
    data_logistic.findFit()
    data_logistic.plot_curve()

# callLogistic_poly()
# callskikit_poly()