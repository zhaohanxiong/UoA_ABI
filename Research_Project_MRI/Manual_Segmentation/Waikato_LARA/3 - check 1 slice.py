import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

os.chdir("C:/Users/Administrator/Desktop/temp3")

temp = cv2.imread("025.tif",cv2.IMREAD_GRAYSCALE)
plt.imshow(temp);plt.show()
