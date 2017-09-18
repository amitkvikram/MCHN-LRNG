#!/usr/bin/python3
from __future__ import print_function
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import math

data = pd.read_csv('train.csv')
#print(data);
# print(data.head());
#
data= data.drop(['Name','PassengerId','Ticket','Cabin'],axis=1);
# print(data.head());
# print(type(data));
# print(data.columns.values)
# print(data.Sex)
data=[data];

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
    print("amit")
