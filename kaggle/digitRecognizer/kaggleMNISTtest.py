import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import linear_model
import cv2

class DigitRecognizer:
    def __init__(self, fname):
        raw_data = open(fname, 'rt')
        data = np.loadtxt(raw_data, delimiter=",", skiprows=1)
        self.X = data
        self.X = np.column_stack((np.ones((self.X.shape[0],)), self.X))
        self.Y = data[:,0].reshape(self.X.shape[0], )
        self.theta = np.loadtxt("parameter.txt", delimiter=',')
        print(self.theta.shape)

    def hypothesis(self):
        # print(self.temp1)
        return 1/(1+(np.exp(-np.dot(self.X, self.theta))))

    def predict(self):
        probability = self.hypothesis()
        max_probability = (np.argmax(probability, axis=1).astype(np.int8))
        print(max_probability.shape)
        image_id = np.arange(1,max_probability.shape[0]+1)
        max_probability1 = np.column_stack((image_id, max_probability))
        # np.savetxt("prediction.txt", max_probability, delimiter=',', newline='\n',fmt='%i')
        np.savetxt('prediction.csv', max_probability1, header='ImageId, Label', delimiter=',', fmt='%i',newline='\n', comments='')

def call_digit_recognizer():
    data_digit= DigitRecognizer("/home/amit/Downloads/test.csv")
    data_digit.predict()


call_digit_recognizer()