import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import linear_model
import cv2

class DigitRecognizer:
    def __init__(self,fname):
        data = scipy.io.loadmat(fname)
        self.X = data['X']
        self.Y = data['y'].reshape(self.X.shape[0], )
        print(self.Y)
        temp = self.X[5,:]
        temp = temp.reshape((20,20)).astype(np.float32)
        cv2.imshow('temp', temp)
        if cv2.waitKey(8000) == 27:
            cv2.destroyAllWindows()

        self.theta = np.zeros((self.X.shape[1]+1,11))
        self.regr = linear_model.LogisticRegression(solver='newton-cg')

    def find_parameter(self):
        for i in range (1,11):
            temp = (self.Y == i).astype(np.float64)
            self.regr.fit(self.X, temp)
            self.theta1 = self.regr.coef_
            intercept = self.regr.intercept_
            self.theta[:,i] = np.column_stack((intercept, self.theta1))
            self.score = self.regr.score(self.X, temp)
            print(self.score)
        print(self.theta.shape)

    def hypothesis(self, theta):
        # print(self.temp1)
        return 1/(1+(np.exp(-np.dot(self.temp1, theta))))

    def maxProbability(self):
        max = 0.0;
        index=1;
        for i in range(1,11):
            probability = self.hypothesis(self.theta[:,i])

            if probability > max:
                max = probability
                index = i;
        return index

    def predict(self):
        for i in range(2000,2110):
            temp = self.X[i, :]
            self.temp1 = np.ones((401, ))
            self.temp1[1:,] = temp
            num = self.maxProbability()
            temp = temp.reshape((20, 20)).astype(np.float32)
            cv2.imshow('image', temp)
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
            print(num, self.Y[i,], '\n')



def call_digit_recognizer():
    data_digit= DigitRecognizer("ex3data1.mat")
    data_digit.find_parameter()
    data_digit.predict()


call_digit_recognizer()