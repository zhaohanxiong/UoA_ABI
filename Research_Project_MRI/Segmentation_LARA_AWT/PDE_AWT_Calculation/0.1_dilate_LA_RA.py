import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

####################################
pat_ID = "00009"
####################################

files = os.listdir(pat_ID+"/AWT")

for i in range(len(files)):

    img = cv2.imread(pat_ID+"/AWT/"+files[i],cv2.IMREAD_GRAYSCALE)

    RA_endo = np.uint8(img==3)
    RA_endo = cv2.dilate(RA_endo,np.ones((2,2)),iterations=1)
    img[RA_endo==1] = 3
    
    #LA_endo = np.uint8(img==4)
    #LA_endo = cv2.dilate(LA_endo,np.ones((2,2)),iterations=1)
    #img[LA_endo==1] = 4
    
    cv2.imwrite(pat_ID+"/AWT/"+files[i],img)
