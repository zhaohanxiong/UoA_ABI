import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

####################################
pat_ID = "00007v2"
check_first = False
####################################

files = os.listdir(pat_ID+"/AWT")

if check_first:
    img = cv2.imread(pat_ID+"/AWT/"+files[30],cv2.IMREAD_GRAYSCALE)
    plt.imshow(img)
    plt.show()
    sys.exit()

for i in range(len(files)):

    img = cv2.imread(pat_ID+"/AWT/"+files[i],cv2.IMREAD_GRAYSCALE)
    img[img==150]  = 1 # RA wall
    img[img==226] = 2 # LA wall
    img[img==76] = 3 # RA cavity
    img[img==179]  = 4 # LA cavity
    
    img[img>4] = 0

    cv2.imwrite(pat_ID+"/AWT/"+files[i],img)
