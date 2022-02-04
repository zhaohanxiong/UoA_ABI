import os
import cv2
import scipy.io
import numpy as np
import pandas as pd
from scipy.ndimage import morphology,label

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
os.chdir("C:/Users/Administrator/Desktop/UKBiobank_LA_2CH_Model+Results_600test")
os.mkdir("Images")
os.mkdir("Images/cmr")
os.mkdir("Images/true")
os.mkdir("Images/pred")

# list files
files = os.listdir("Prediction Sample")

# store output scores
column_headers = ["DSC","sty","spy","IoU",
				  "pred_dia","true_dia","dia_err","dia_per",
				  "pred_vol","true_vol","vol_err","vol_per", 
				  "s2s_dist"]

LAendo_score  = np.zeros([len(files),len(column_headers)])

# loop through all files
for n in range(len(files)):
	
	print("Processing "+"{0:03}".format(n+1))
	
	# read data
	dat = scipy.io.loadmat("Prediction Sample/"+files[n])
	true,pred,cmr = dat["true"].astype(np.uint8),dat["pred"].astype(np.uint8),dat["cmr"].astype(np.uint8)

	cv2.imwrite("Images/cmr/img"+str(n)+".tif",cmr)
	cv2.imwrite("Images/true/img"+str(n)+".tif",true)
	cv2.imwrite("Images/pred/img"+str(n)+".tif",pred)
	
	if True:
		## ----------------------------------------------------------------------------------------------- LA ENDO
		t,p = true>=1,pred>=1
		
		if np.sum(t) == 0 or np.sum(p) == 0:
			
			LAendo_score[n,:] = 1
			
		else:

			tp,fp = np.sum((p == 1) * (p == t)),np.sum((p == 1) * (p != t))
			fn,tn = np.sum((p == 0) * (p != t)),np.sum((p == 0) * (p == t))
			
			LAendo_score[n,0] = 2 * np.sum(p * t) / (np.sum(p) + np.sum(t))	# Dice score
			LAendo_score[n,1] = tp / (tp + fn)								# Sensitivity
			LAendo_score[n,2] = tn / (tn + fp)								# Specificity
			LAendo_score[n,3] = np.sum(p * t) / np.sum((p + t) > 0)			# Intersection over union

			# LA diameter
			p_dia,t_dia  = np.max(np.sum(p,1))*1.8,np.max(np.sum(t,1))*1.8
			LAendo_score[n,4] = p_dia										# predicted diameter
			LAendo_score[n,5] = t_dia										# true diameter
			LAendo_score[n,6] = np.abs(t_dia - p_dia)						# diameter error in pixels
			LAendo_score[n,7] = np.abs(t_dia - p_dia) / t_dia				# diameter error in percentage		
			
			# LA volume
			p_vol,t_vol  = np.sum(p)*1.8*1.8,np.sum(t)*1.8*1.8
			LAendo_score[n,8]  = p_vol										# predicted volume
			LAendo_score[n,9]  = t_vol										# true volume
			LAendo_score[n,10] = np.abs(t_vol - p_vol)						# volume error in pixels
			LAendo_score[n,11] = np.abs(t_vol - p_vol) / t_vol				# volume error in percentage

			# surface to surface distance: mean (average symetric surface2surface dist) max (symetric Hausdorff dist) 
			LAendo_score[n,12] = surfd(p,t,[1.8,1.8],1).mean()
		
# Save data to csv output
pd.DataFrame(LAendo_score).to_csv("LAendo.csv",header=column_headers,index=False)

# summarize
print("\n\n---------------------- LA Endo Evaluation")
print("Dice Score: "+str(np.mean(LAendo_score[:,0])))
print("      S2SD: "+str(np.mean(LAendo_score[:,12])))
