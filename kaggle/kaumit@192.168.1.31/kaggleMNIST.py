import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import linear_model
import cv2

class DigitRecognizer:
    def __init__(self, fname):
        raw_data = open(fname, 'rt')
        # print(next(raw_data))
        data = np.loadtxt(raw_data, delimiter=",", skiprows=1)
        self.X = data[:21000,1:]
        self.X_test = data[21000:,1:]
        self.Y = data[:21000,0].reshape(self.X.shape[0], )
        self.Y_test = data[21000:,0]
        self.theta = np.zeros((self.X.shape[1]+1,10))
        self.regr = linear_model.LogisticRegression(solver='lbfgs')

    def find_parameter(self):
        for i in range(0, 10):
            print(i)
            temp = (self.Y == i).astype(np.float64)
            temp_test = (self.Y_test == i).astype(np.float64)
            self.regr.fit(self.X, temp)
            self.theta1 = self.regr.coef_
            intercept = self.regr.intercept_
            self.theta[:,i] = np.column_stack((intercept, self.theta1))
            self.score = self.regr.score(self.X_test, temp_test)
            print(self.score)
        print(np.show_config())
        np.savetxt("parameter.txt",self.theta,delimiter=',',newline='\n')

    def hypothesis(self, theta):
        # print(self.temp1)
        return 1/(1+(np.exp(-np.dot(self.temp1, theta))))



def call_digit_recognizer():
    data_digit= DigitRecognizer("train.csv")
    data_digit.find_parameter()
    # data_digit.predict()


call_digit_recognizer()