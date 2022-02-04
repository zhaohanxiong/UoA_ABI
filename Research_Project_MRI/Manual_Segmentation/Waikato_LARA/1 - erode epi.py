import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

os.chdir("C:/Users/Administrator/Desktop")

files = os.listdir("temp") # LA and RA epi masks from amira

os.mkdir("temp2")

mask = np.zeros([640,640,44])

for i in range(len(files)):

    mask[:,:,i] = cv2.imread("temp/"+files[i],cv2.IMREAD_GRAYSCALE)

    RA = np.uint8(mask[:,:,i] == 179)
    LA = np.uint8(mask[:,:,i] == 150)

    RAendo = cv2.erode(RA,np.ones((3,3),np.uint8),iterations = 6)
    LAendo = cv2.erode(LA,np.ones((3,3),np.uint8),iterations = 4)

    out = np.zeros([640,640]).astype(np.uint8)
    out[RA == 1] = 1
    out[LA == 1] = 2
    out[RAendo == 1] = 3
    out[LAendo == 1] = 4

    cv2.imwrite("temp2/"+"{0:03}".format(i+1)+".tif",out)