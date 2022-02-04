import os
import cv2
import scipy.io
import numpy as np
from scipy import ndimage

# set directory location
os.chdir("C:/Users/Administrator/Desktop/UKBiobank_LA_2CH_Model+Results_600test")
#os.chdir("C:/Users/Administrator/Desktop/UKBiobank_LA_4CH_Model+Results_600test")

# list files
files = os.listdir("Prediction Sample")

MSE,x_err,y_err = [],[],[]

# loop through all files
for n in range(len(files)):
	
	print("Processing "+"{0:03}".format(n+1))
	
	# read data
	dat = scipy.io.loadmat("Prediction Sample/"+files[n])
	true,pred,cmr = dat["true"].astype(np.uint8),dat["pred"].astype(np.uint8),dat["cmr"].astype(np.uint8)

	# find positive pixels
	t,p = true>=1,pred>=1
	
	if np.sum(t) == 0 or np.sum(p) == 0:
		MSE.append(0)
	else:

		# compute COM
		true = ndimage.measurements.center_of_mass(true > 0)
		pred = ndimage.measurements.center_of_mass(pred > 0)
		
		# compute mse
		MSE.append(np.sqrt((true[0] - pred[0])**2 + (true[1] - pred[1])**2))
		x_err.append(np.abs(true[0] - pred[0]))
		y_err.append(np.abs(true[1] - pred[1]))

MSE,x_err,y_err = np.array(MSE)*1.8,np.array(x_err)*1.8,np.array(y_err)*1.8

# summarize
print("\n\n---------------------- ROI Evaluation")
print("MSE: "+str(round(np.mean(MSE),2))+" ("+str(round(np.std(MSE),2))+")")
print("X Error: "+str(round(np.mean(x_err),2))+" ("+str(round(np.std(x_err),2))+")")
print("Y Error: "+str(round(np.mean(y_err),2))+" ("+str(round(np.std(y_err),2))+")")