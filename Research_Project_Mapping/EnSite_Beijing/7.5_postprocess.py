import os
import cv2
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.stats import ks_2samp
from scipy.ndimage import morphology,label
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics.pairwise import pairwise_distances

# surface to surface distance (https://mlnotebook.github.io/post/surface-distance-function/)
def surfd(input1, input2, sampling=1, connectivity=1):

	input_1 = np.atleast_1d(input1.astype(np.bool))
	input_2 = np.atleast_1d(input2.astype(np.bool))

	conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

	S      = input_1.astype(np.uint32) - morphology.binary_erosion(input_1, conn)
	Sprime = input_2.astype(np.uint32) - morphology.binary_erosion(input_2, conn)

	dta = morphology.distance_transform_edt(S,sampling)
	dtb = morphology.distance_transform_edt(Sprime,sampling)
	
	sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])

	return sds

# set directory location
os.chdir("C:/Users/zxio506/Desktop/Catheter Path Test Set")

# list files
files = os.listdir("Prediction Sample")

# make new directories to save outputs
if False: # True False
	os.mkdir("Image Stack")

# store output scores
column_headers = ["DSC","sty","spy","IoU",
				  "pred_dia","true_dia","dia_err","dia_per",
				  "pred_vol","true_vol","vol_err","vol_per", 
				  "s2s_dist","coverage"]

LA_endo  = np.zeros([len(files),len(column_headers)])

# loop through all files
for n in range(len(files)):
	
	print("Processing "+files[n])
	
	# read data
	dat = scipy.io.loadmat("Prediction Sample/"+files[n])
	true,pred,in_dat = dat["true"].astype(np.uint8),dat["pred"].astype(np.uint8),dat["input"]

	### Write Output to Image Stacks -----------------------------------------------------------------------------
	if False: # True False
	
		# write to image stack
		os.mkdir("Image Stack/"+files[n])
		os.mkdir("Image Stack/"+files[n]+"/true")
		os.mkdir("Image Stack/"+files[n]+"/pred")
		os.mkdir("Image Stack/"+files[n]+"/input")
		
		for i in range(true.shape[2]):
			img_name = "{0:03}".format(i+1)+".tif"
			cv2.imwrite("Image Stack/"+files[n]+"/true/"+img_name,true[:,:,i])
			cv2.imwrite("Image Stack/"+files[n]+"/pred/"+img_name,pred[:,:,i])
			cv2.imwrite("Image Stack/"+files[n]+"/input/"+img_name,in_dat[:,:,i,0]*255)

	### Evaluate Metrics -----------------------------------------------------------------------------------------
	if True: # True False

		# ----------------------------------------------------------------------------------------------- LA ENDO
		t,p = true==1,pred==1

		tp,fp = np.sum((p == 1) * (p == t)),np.sum((p == 1) * (p != t))
		fn,tn = np.sum((p == 0) * (p != t)),np.sum((p == 0) * (p == t))
		
		LA_endo[n,0] = 2 * np.sum(p * t) / (np.sum(p) + np.sum(t))	# Dice score
		LA_endo[n,1] = tp / (tp + fn)								# Sensitivity
		LA_endo[n,2] = tn / (tn + fp)								# Specificity
		LA_endo[n,3] = np.sum(p * t) / np.sum((p + t) > 0)			# Intersection over union

		# LA diameter
		p_dia,t_dia  = np.max(np.sum(p,0))*0.625,np.max(np.sum(t,0))*0.625
		LA_endo[n,4] = p_dia										# predicted diameter
		LA_endo[n,5] = t_dia										# true diameter
		LA_endo[n,6] = np.abs(t_dia - p_dia)						# diameter error in mm
		LA_endo[n,7] = np.abs(t_dia - p_dia) / t_dia				# diameter error in percentage		
		
		# LA volume
		p_vol,t_vol  = np.sum(p)*0.625*0.625*2.5,np.sum(t)*0.625*0.625*2.5
		LA_endo[n,8]  = p_vol										# predicted volume
		LA_endo[n,9]  = t_vol										# true volume
		LA_endo[n,10] = np.abs(t_vol - p_vol)						# volume error in mm
		LA_endo[n,11] = np.abs(t_vol - p_vol) / t_vol				# volume error in percentage

		# surface to surface distance: mean (average symetric surface2surface dist) max (symetric Hausdorff dist) 
		LA_endo[n,12] = surfd(p,t,[0.625,0.625,2.5],1).mean()
		
		# coverage statistics for the sample
		true_erode = np.zeros_like(t)
		
		for i in range(true_erode.shape[2]):
			true_erode[:,:,i] = cv2.erode(np.uint8(t[:,:,i]),np.ones((3,3),np.uint8),iterations=2)

		true_shell = true - true_erode
		
		in_dat[in_dat>0] = 1
		in_dat           = in_dat[true_shell==1]
		
		LA_endo[n,13] = np.sum(in_dat)/np.sum(true_shell)
		
# Save data to csv output
pd.DataFrame(LA_endo).to_csv("Test_Evaluation.csv",header=column_headers,index=False)

# summarize
print("\n\n---------------------- LA Reconstruction From Catheter Path")
print("Dice Score: "+str(np.mean(LA_endo[:,0])))
print("")
print("Dice Score Utah:    "   + str(np.mean(LA_endo[2:-4,0])))
print("Dice Score Waikato: "   + str(np.mean(LA_endo[-4:,0])))
print("Dice Score Melbourne: " + str(np.mean(LA_endo[:2,0])))
print("S2S Distance:       "   + str(np.mean(LA_endo[:,12])))
