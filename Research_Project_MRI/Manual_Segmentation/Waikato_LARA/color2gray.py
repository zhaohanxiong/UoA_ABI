import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

root = "test025.mat/pred"

files = os.listdir(root)
for i in range(len(files)):

        temp = cv2.imread(root+"/"+files[i],cv2.IMREAD_GRAYSCALE)

        if (i >= 32):
                temp[temp==2] = 1

        cv2.imwrite(root+"/"+files[i],temp)
