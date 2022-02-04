import os
import cv2
import scipy.io
import numpy as np
import pandas as pd
from scipy import ndimage

# set directory location
os.chdir("C:/Users/Administrator/Desktop/UtahWaikato Test Set")

# list files
files = os.listdir("Prediction Sample")

MSE,x_err,y_err = [],[],[]

# loop through all files
for n in range(len(files)):
	
	print("Processing "+files[n])
	
	# read data
	dat = scipy.io.loadmat("Prediction Sample/"+files[n])
	true,pred = dat["true"].astype(np.uint8),dat["pred"].astype(np.uint8)
	
	# post-process prediction
	for i in range(true.shape[2]):

		# compute COM
		slice_true = ndimage.measurements.center_of_mass(true[:,:,i] > 0)
		slice_pred = ndimage.measurements.center_of_mass(pred[:,:,i] > 0)
		
		# compute mse
		MSE.append(np.sqrt((slice_true[0] - slice_pred[0])**2 + (slice_true[1] - slice_pred[1])**2))
		x_err.append(np.abs(slice_true[0] - slice_pred[0]))
		y_err.append(np.abs(slice_true[1] - slice_pred[1]))

MSE,x_err,y_err = np.array(MSE)*0.625,np.array(x_err)*0.625,np.array(y_err)*0.625

# summarize
print("\n\n---------------------- ROI Evaluation")
print("MSE: "+str(round(np.nanmean(MSE),2))+" ("+str(round(np.nanstd(MSE),2))+")")
print("X Error: "+str(round(np.nanmean(x_err),2))+" ("+str(round(np.nanstd(x_err),2))+")")
print("Y Error: "+str(round(np.nanmean(y_err),2))+" ("+str(round(np.nanstd(y_err),2))+")")