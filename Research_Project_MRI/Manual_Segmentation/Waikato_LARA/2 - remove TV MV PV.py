import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

os.chdir("C:/Users/Administrator/Desktop")

files1 = os.listdir("temp1") # remove pixels saved from amira files
files2 = os.listdir("temp2") # prior files saved from first script
os.mkdir("temp3")

mask1,mask2 = np.zeros([640,640,44]),np.zeros([640,640,44])

for i in range(len(files1)):

    mask1[:,:,i] = cv2.imread("temp1/"+files1[i],cv2.IMREAD_GRAYSCALE)
    mask2[:,:,i] = cv2.imread("temp2/"+files2[i],cv2.IMREAD_GRAYSCALE)

    remove = mask1[:,:,i] == 29

    out = mask2[:,:,i]
    out[remove] = 0

    cv2.imwrite("temp3/"+"{0:03}".format(i+1)+".tif",out.astype(np.uint8))