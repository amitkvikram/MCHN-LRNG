#!/usr/bin/python3.5
#Programme: Implementation of Linear Regression using standard python3 library
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
import scipy

class uniVariate_grad:

    def __init__(self):
        self.data= np.loadtxt("linearReg_m.txt",delimiter=',');
        self.X= self.data[:,0:-1];
        self.Y= self.data[:,1:2];
        self.theta= np.zeros((self.X.shape[1]+1,1));
    def iterAlpha(self, alpha):
        self.alpha=alpha;
        self.J=[];
    def addX0(self):
        one = np.ones((self.X.shape[0],1));
        self.X= np.column_stack((one,self.X));
    def plotData(self):
        plt.plot(list(self.X), list(self.Y), 'rx');
        plt.xlabel('xAxis');
        plt.ylabel('yAxis');
        plt.title("singleVariable Data")
        plt.show();
    def costCalc(self):
        J= (1/(2*self.m))*(np.sum(np.square(np.dot(self.X,self.theta)-self.Y)));
        self.J.append(J);
    def gradientDescent(self,iter1):
        self.iter=iter1;
        m=self.m = self.X.shape[0]
        for i in range(iter1):
            self.costCalc();
            self.theta= self.theta- (self.alpha/m)*(np.dot(self.X.T, (np.dot(self.X, self.theta) - self.Y)));

    def plotCost(self):
        xAxis= np.arange(0,self.iter);
        line, =plt.plot(xAxis, self.J)
        plt.xlabel("iteration")
        plt.ylabel("Cost")
        plt.title("Cost vs iteration")
        plt.show();
    def plotCurve(self):
        max=np.max(self.X);
        xAxis= np.arange(0,np.amax(self.X),0.01);
        xAxis= np.reshape(xAxis, (xAxis.shape[0],1));
        xAxis= np.column_stack((np.ones((len(xAxis),1)),xAxis));
        yAxis= np.dot(xAxis, self.theta);
        plt.plot(list(self.X[:,1:]), list(self.Y), 'rx');
        plt.xlabel('xAxis');
        plt.ylabel('yAxis');
        plt.title("singleVariable gradientDescent Prediction")
        plt.plot(list(xAxis[:,1:]), list(yAxis));
        plt.show();
    def printTheta(self):
        J= (1/(2*self.m))*(np.sum(np.square(np.dot(self.X,self.theta)-self.Y)));
        print(J)
        print("theta=",self.theta)

class uni_skikit:
    def __init__(self):
        self.data= np.loadtxt("linearReg_m.txt",delimiter=',');
        self.X= self.data[:,0:-1];
        self.Y= self.data[:,1:2];
    def findFit(self):
        regr= linear_model.LinearRegression();
        regr.fit(self.X, self.Y);
        self.Y_predict= regr.predict(self.X);
        print(regr.score(self.X, self.Y)*((self.Y - self.Y.mean()) ** 2).sum())
    def plotData(self):
        plt.plot(list(self.X), list(self.Y), 'rx');
        plt.xlabel('xAxis');
        plt.ylabel('yAxis');
        plt.title("singleVariable skikit Data")
        plt.show();
    def plotCurve(self):
        plt.plot(list(self.X), list(self.Y), 'rx');
        plt.xlabel('xAxis');
        plt.ylabel('yAxis');
        plt.title("skikit PRediction")
        plt.plot(self.X, self.Y_predict);
        plt.show();



def uni_grad():
    data_uni = uniVariate_grad();
    data_uni.iterAlpha(0.01);          #give alpha as argument
    # data_uni.plotData();
    data_uni.addX0();
    data_uni.gradientDescent(1000);    #give number of iteration as argument
    data_uni.plotCost();
    data_uni.plotCurve();
    data_uni.printTheta();

def skikit_linearReg():
    data_uni1 = uni_skikit();
    data_uni1.findFit();
    data_uni1.plotData();
    data_uni1.plotCurve();
uni_grad();
skikit_linearReg();


###################################################################################################################################

class multiVariate():
        def __init__(self):
            self.data= np.loadtxt("linearReg_multi.txt",delimiter=',');
            self.X= self.data[:,0:-1];
            self.Y= self.data[:,-1:];
            self.X1=self.X  #copy of self.X without normalization
            self.theta= np.zeros((self.X.shape[1]+1,1));
        def iterAlpha(self, alpha):
            self.alpha=alpha;
            self.J=[];
        def addX0(self):
            one = np.ones((self.X.shape[0],1));
            self.X= np.column_stack((one,self.X));
        def plotData(self):
            if self.X1.shape[1]!=2:
                return
            ax= plt.gca(projection="3d",title="3D1",axisbg='gray')
            plt.xlabel('xAxis');
            plt.ylabel('yAxis');
            plt.title("multiVariateData")
            x=self.X[:,0];
            y=self.X[:,1]
            z=self.Y.reshape((self.Y.shape[0],));
            line,=plt.plot(x,y,z,'^' ,color='r')
            plt.show();
        def featureNormalize(self):
            if self.X1.shape[1]!=2:
                return
            mu = np.sum(self.X,0)/self.X.shape[0];
            max1= np.amax(self.X, 0)
            min1= np.amin(self.X,0)
            self.sigma = max1 - min1
            self.X= (self.X)/self.sigma;

        def costCalc(self):

            J= (1/(2*self.m))*(np.sum(np.square(np.dot(self.X,self.theta)-self.Y)));
            self.J.append(J);

        def gradientDescent(self,iter1):
            self.iter=iter1;
            m=self.m = self.X.shape[0]
            for i in range(iter1):
                self.theta=self.theta-(self.alpha/m)*(np.dot(self.X.T, (np.dot(self.X, self.theta) - self.Y)));
                self.costCalc()
        def plotCost(self):
            xAxis= np.arange(0,self.iter);
            line, =plt.plot(xAxis, self.J);
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("cost vs iteration")
            plt.show();

        def plotCurve(self):
            if self.X1.shape[1]!=2:
                return;
            ax = plt.gca(projection='3d')
            max1= np.amax(self.X1[:,1:])
            max2= np.amax(self.X1[:,0:])
            xAxis= np.linspace(0,max2, 120);
            xAxis= np.tile(xAxis, 120).reshape(120,120);
            yAxis= np.linspace(0,max1, 120);
            yAxis= np.tile(yAxis, 120).reshape(120,120);
            yAxis=yAxis.T
            np.column_stack((np.ones((120,)),xAxis))
            Z= np.empty_like(xAxis);
            for i in range(120):
                Z[:,i]= np.dot(np.column_stack((np.ones((120,)),xAxis[:,i]/self.sigma[0,],yAxis[:,i]/self.sigma[1,])),self.theta).reshape(120,)
            ax.plot_surface(xAxis,yAxis, Z, rstride=1, cstride=1)
            plt.xlabel('xAxis')
            plt.ylabel('yAxis')
            plt.xlabel('xAxis');
            plt.ylabel('yAxis');
            plt.title("Gradient Prediction")
            x=self.X1[:,0];
            y=self.X1[:,1]
            z=self.Y.reshape((self.Y.shape[0],));
            line,=plt.plot(x,y,z,'^' ,color='r')
            plt.show()
        def printTheta(self):
            print("theta=",self.theta)

class multi_skikit():
    def __init__(self):
        self.data= np.loadtxt("linearReg_multi.txt",delimiter=',');
        self.X= self.data[:,0:-1];
        self.Y= self.data[:,-1:];
    def findFit(self):
        regr= linear_model.LinearRegression(normalize = True);
        regr.fit(self.X, self.Y);
        return regr;
    def plotCurve(self):
        if self.X.shape[1]!=2:
            return;
        max1= np.amax(self.X[:,1:])
        max2= np.amax(self.X[:,0:])
        ax = plt.gca(projection='3d')
        xAxis= np.linspace(0,max2, 120);
        xAxis= np.tile(xAxis, 120).reshape(120,120);
        yAxis= np.linspace(0,max1, 120);
        yAxis= np.tile(yAxis, 120).reshape(120,120);
        yAxis=yAxis.T
        print(xAxis)
        print(yAxis)
        Z= np.empty_like(xAxis);
        for i in range(120):
            Z[:,i]= self.regr.predict(np.column_stack((xAxis[:,i],yAxis[:,i]))).reshape(120,);
        ax.plot_surface(xAxis,yAxis, Z, rstride=1, cstride=1)
        plt.xlabel('xAxis')
        plt.ylabel('yAxis')
        plt.xlabel('xAxis');
        plt.ylabel('yAxis');
        plt.title("Skikit Prediction")
        x=self.X[:,0];
        y=self.X[:,1]
        z=self.Y.reshape((self.Y.shape[0],));
        line,=plt.plot(x,y,z,'^' ,color='r')
        plt.show()

def multi_grad():
    data_multi = multiVariate();
    data_multi.iterAlpha(0.03);          #give alpha as argument
    data_multi.plotData();
    data_multi.featureNormalize();
    data_multi.addX0();
    data_multi.gradientDescent(3000);    #give number of iteration as argument
    data_multi.plotCost();
    data_multi.plotCurve();
    data_multi.printTheta();
def skikit_linearReg1():
    data_multi1 = multi_skikit();
    data_multi1.regr=data_multi1.findFit();
    # data_uni1.plotData();
    data_multi1.plotCurve();
multi_grad();
skikit_linearReg1();
