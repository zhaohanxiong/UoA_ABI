import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

####################################
pat_ID = "00007v2"
####################################

files = os.listdir(pat_ID+"/AWT")

for i in range(len(files)):

    img = cv2.imread(pat_ID+"/AWT/"+files[i],cv2.IMREAD_GRAYSCALE)

    #RA_endo = np.uint8(img==3)
    #RA_endo_new = cv2.erode(RA_endo,np.ones((2,2)),iterations=1)
    #overlay = RA_endo - RA_endo_new
    #img[overlay==1] = 1
    
    LA_endo = np.uint8(img==4)
    LA_endo_new = cv2.erode(LA_endo,np.ones((2,2)),iterations=1)
    overlay = LA_endo - LA_endo_new
    img[overlay==1] = 2
    
    cv2.imwrite(pat_ID+"/AWT/"+files[i],img)
