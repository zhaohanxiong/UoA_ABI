import numpy as np
import cv2
import os

def color2gray(file):
	#file = "CARMA_1401_pre_NewSeg_w_Septum_014.tif"
	temp = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
	cv2.imwrite(file,temp)


root = "1387 4mo/CARMA_1387_4mo_full"
root = "LAendoNoVeins"
files = os.listdir(root)
for i in range(len(files)):
	color2gray(root+"/"+files[i])
