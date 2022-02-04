import os
import cv2
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.stats import ks_2samp, pearsonr
from scipy.ndimage import morphology,label
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

def utah_threshold(img, lab, std_factor):
	
	# Inputs: img = 3D LGE-MRI, lab = 3D LA wall label, std_factor = tune threshold (2-4)
	
	# leave the LA wall pixels only
	img[lab!=1] = 0
	
	# limit intensity ranges to only 2% - 40%
	thresh_low  = np.quantile(img[img>0], 0.02)
	thresh_high = np.quantile(img[img>0], 0.40)
	
	img_norm = np.copy(img)
	img_norm[img_norm < thresh_low]  = 0
	img_norm[img_norm > thresh_high] = 0
	
	# define fibrosis output map
	fibrosis = np.zeros_like(img)
	
	# loop through each slice
	for i in range(img.shape[2]):		
		if np.sum(img[:,:,i]) > 0:

			# define individual slice images
			tmp,fibrosis_slice = img_norm[:,:,i],fibrosis[:,:,i]
			
			# define threshold
			thres_slice = np.mean(tmp[tmp>0]) + np.std(tmp[tmp>0]) * std_factor

			# add to fibrosis slice
			fibrosis_slice[img[:,:,i] > thres_slice] = 1
			fibrosis[:,:,i] = fibrosis_slice
		
	return(fibrosis)

# set directory location
os.chdir("C:/Users/zxio506/Desktop/UtahWaikato Test Set")
#os.chdir("C:/Users/Administrator/Desktop/UtahWaikato Wall Fibrosis Test")

# list files
files = os.listdir("Prediction Sample")

# make new directories to save outputs
if True: # True False
	os.mkdir("Image Stack Fibrosis")
	os.mkdir("Fibrosis Distributions")

# store output scores
column_headers = ["DSC","sty","spy","IoU","s2s_dist","SNR",
				  "fib_true","fib_pred","fib_err",
				  "KS_d","KS_p",
				  "PCC"]
LA_FIBR        = np.zeros([len(files),len(column_headers)])
RA_FIBR        = np.zeros([len(files),len(column_headers)])

# loop through all files
for n in range(len(files)):
	
	print("Processing "+files[n])
	
	# read data (0 = background, 1 = LARA wall, 2 = RA endo, 3 = LA endo)
	dat = scipy.io.loadmat("Prediction Sample/"+files[n])
	true,pred = dat["true"].astype(np.uint8),dat["pred"].astype(np.uint8)
	
	# process lgemri data
	lgemri = dat["lgemri"]
	lgemri = lgemri/np.max(lgemri)*255
	lgemri = lgemri.astype(np.uint8)

	# adjust brightness of 2 waiakto samples that suck ass
	if any([files[n] == s for s in ["newtest004.mat","newtest006.mat"]]):
		temp = lgemri[:,:,36:]*2.0
		temp[temp>255] = 255
		lgemri[:,:,36:] = temp
	
	# seperate LA and RA walls
	true_RA_dil,pred_RA_dil = np.zeros_like(lgemri),np.zeros_like(lgemri)
	
	for i in range(lgemri.shape[2]):
		true_RA_dil[:,:,i] = cv2.dilate(np.uint8(true[:,:,i]==2),np.ones((3,3),np.uint8),iterations=7)
		pred_RA_dil[:,:,i] = cv2.dilate(np.uint8(pred[:,:,i]==2),np.ones((3,3),np.uint8),iterations=7)
	
	# get RA wall and LA wall individually
	true_RA_wall,true_LA_wall                             = np.zeros_like(lgemri),np.zeros_like(lgemri)
	true_RA_wall[np.logical_and(true==1,true_RA_dil==1)]  = 1
	true_LA_wall[np.logical_and(true==1,true_RA_wall!=1)] = 1
	
	pred_RA_wall,pred_LA_wall                             = np.zeros_like(lgemri),np.zeros_like(lgemri)
	pred_RA_wall[np.logical_and(pred==1,pred_RA_dil==1)]  = 1
	pred_LA_wall[np.logical_and(pred==1,pred_RA_wall!=1)] = 1
	
	# seperate fibrosis into left and right (4 = RA fibrosis, 5 = LA fibrosis)
	if "new" in files[n]:
		LA_thres = 4.25
		RA_thres = 4
	else:
		LA_thres = 3.85
		RA_thres = 4.25
	
	fib_true_RA = utah_threshold(np.copy(lgemri), true_RA_wall, RA_thres)
	fib_true_LA = utah_threshold(np.copy(lgemri), true_LA_wall, LA_thres)
	
	fib_pred_RA = utah_threshold(np.copy(lgemri), pred_RA_wall, RA_thres)
	fib_pred_LA = utah_threshold(np.copy(lgemri), pred_LA_wall, LA_thres)

	# add fibrosis to label
	true[fib_true_RA==1] = 4
	true[fib_true_LA==1] = 5
	
	pred[fib_pred_RA==1] = 4
	pred[fib_pred_LA==1] = 5
	
	### Write Output to Image Stacks -----------------------------------------------------------------------------
	if True: # True False
	
		write_true = np.copy(true)		
		write_true[write_true==2] = 0
		write_true[write_true==3] = 0
		write_true[write_true==4] = 2
		write_true[write_true==5] = 3
		
		write_pred = np.copy(pred)		
		write_pred[write_pred==2] = 0
		write_pred[write_pred==3] = 0
		write_pred[write_pred==4] = 2
		write_pred[write_pred==5] = 3
		
		# write to image stack
		os.mkdir("Image Stack Fibrosis/"+files[n])
		os.mkdir("Image Stack Fibrosis/"+files[n]+"/true")
		os.mkdir("Image Stack Fibrosis/"+files[n]+"/pred")
		os.mkdir("Image Stack Fibrosis/"+files[n]+"/lgemri")
		
		for i in range(true.shape[2]):
			img_name = "{0:03}".format(i+1)+".tif"
			cv2.imwrite("Image Stack Fibrosis/"+files[n]+"/true/"+img_name,write_true[:,:,i])
			cv2.imwrite("Image Stack Fibrosis/"+files[n]+"/pred/"+img_name,write_pred[:,:,i])
			cv2.imwrite("Image Stack Fibrosis/"+files[n]+"/lgemri/"+img_name,lgemri[:,:,i])

	### Visualize Fibrosis Distribution in X/Y/Z ----------------------------------------------------------------
	if True: # True False
	
		# ### LA ###############################################################################
		# compute distribution of true and predicted fibrosis
		t_com = np.array(ndimage.measurements.center_of_mass(true == 3))
		p_com = np.array(ndimage.measurements.center_of_mass(pred == 3))
		
		t_pts,p_pts = np.where(true == 5), np.where(pred == 5)
		t_pts,p_pts = np.array(t_pts).T, np.array(p_pts).T
		t_pts,p_pts = t_pts - t_com, p_pts - p_com
		
		# generate plot
		plt.figure(figsize=(5,10))
		
		plt.subplot(311)
		plt.title("Fibrosis Distribution in X Axis")
		plt.hist(t_pts[:,0],40,alpha=0.75,edgecolor='black',linewidth=0.5,color="deepskyblue")
		plt.hist(p_pts[:,0],40,alpha=0.5,edgecolor='black',linewidth=0.5,color="tomato")
		
		plt.subplot(312)
		plt.title("Fibrosis Distribution in Y Axis")
		plt.hist(t_pts[:,1],40,alpha=0.75,edgecolor='black',linewidth=0.5,color="deepskyblue")
		plt.hist(p_pts[:,1],40,alpha=0.5,edgecolor='black',linewidth=0.5,color="tomato")
		
		plt.subplot(313)
		plt.title("Fibrosis Distribution in Z Axis")
		plt.hist(t_pts[:,2],20,alpha=0.75,edgecolor='black',linewidth=0.5,color="deepskyblue")
		plt.hist(p_pts[:,2],20,alpha=0.5,edgecolor='black',linewidth=0.5,color="tomato")
		
		plt.savefig("Fibrosis Distributions/LA_"+files[n]+".png")
		plt.close()
		
		# ### RA ###############################################################################
		# compute distribution of true and predicted fibrosis
		t_com = np.array(ndimage.measurements.center_of_mass(true == 2))
		p_com = np.array(ndimage.measurements.center_of_mass(pred == 2))
		
		t_pts,p_pts = np.where(true == 4), np.where(pred == 4)
		t_pts,p_pts = np.array(t_pts).T, np.array(p_pts).T
		t_pts,p_pts = t_pts - t_com, p_pts - p_com
		
		# generate plot
		plt.figure(figsize=(5,10))
		
		plt.subplot(311)
		plt.title("Fibrosis Distribution in X Axis")
		plt.hist(t_pts[:,0],40,alpha=0.75,edgecolor='black',linewidth=0.5,color="deepskyblue")
		plt.hist(p_pts[:,0],40,alpha=0.5,edgecolor='black',linewidth=0.5,color="tomato")
		
		plt.subplot(312)
		plt.title("Fibrosis Distribution in Y Axis")
		plt.hist(t_pts[:,1],40,alpha=0.75,edgecolor='black',linewidth=0.5,color="deepskyblue")
		plt.hist(p_pts[:,1],40,alpha=0.5,edgecolor='black',linewidth=0.5,color="tomato")
		
		plt.subplot(313)
		plt.title("Fibrosis Distribution in Z Axis")
		plt.hist(t_pts[:,2],20,alpha=0.75,edgecolor='black',linewidth=0.5,color="deepskyblue")
		plt.hist(p_pts[:,2],20,alpha=0.5,edgecolor='black',linewidth=0.5,color="tomato")
		
		plt.savefig("Fibrosis Distributions/RA_"+files[n]+".png")
		plt.close()

	### Evaluate Metrics -----------------------------------------------------------------------------------------
	if True: # True False

		# ----------------------------------------------------------------------------------------------- RA FIBROSIS
		t,p = true==4,pred==4

		tp,fp = np.sum((p == 1) * (p == t)),np.sum((p == 1) * (p != t))
		fn,tn = np.sum((p == 0) * (p != t)),np.sum((p == 0) * (p == t))
		
		RA_FIBR[n,0] = 2 * np.sum(p * t) / (np.sum(p) + np.sum(t))	# Dice score
		RA_FIBR[n,1] = tp / (tp + fn)								# Sensitivity
		RA_FIBR[n,2] = tn / (tn + fp)								# Specificity
		RA_FIBR[n,3] = np.sum(p * t) / np.sum((p + t) > 0)			# Intersection over union

		# surface to surface distance: mean (average symetric surface2surface dist) max (symetric Hausdorff dist) 
		RA_FIBR[n,4] = surfd(p,t,[0.625,0.625,2.5],1).mean()
		
		# SNR
		RA_FIBR[n,5] = np.abs(np.mean(lgemri[true == 2]) - np.mean(lgemri[true == 0])) / np.std(lgemri[true == 0])
		
		# fibrosis percentage
		RA_FIBR[n,6] = np.sum(t)/np.sum(true_RA_wall)
		RA_FIBR[n,7] = np.sum(p)/np.sum(pred_RA_wall)
		RA_FIBR[n,8] = np.abs(RA_FIBR[n,6] - RA_FIBR[n,7])*100

		# 2D Kolmogorov-Smirnov statistic
		t_com,p_com             = ndimage.measurements.center_of_mass(true == 2),ndimage.measurements.center_of_mass(pred == 2)
		t_pts,p_pts             = np.where(t),np.where(p)
		t_com,p_com,t_pts,p_pts = np.array(t_com),np.array(p_com),np.array(t_pts).T,np.array(p_pts).T
		t_pts,p_pts             = t_pts-t_com, p_pts-p_com
		
		KS_x,KS_y,KS_z = ks_2samp(t_pts[:,0],p_pts[:,0]),ks_2samp(t_pts[:,1],p_pts[:,1]),ks_2samp(t_pts[:,2],p_pts[:,2])
		RA_FIBR[n,9]   = 1 - np.mean([KS_x[0],KS_y[0],KS_z[0]])
		RA_FIBR[n,10]  = np.min([KS_x[1],KS_y[1],KS_z[1]])
		
		# correlation coefficients
		#RA_FIBR[n,11] = pearsonr(t,p)[0]
		
		# ----------------------------------------------------------------------------------------------- LA FIBROSIS
		t,p = true==5,pred==5

		tp,fp = np.sum((p == 1) * (p == t)),np.sum((p == 1) * (p != t))
		fn,tn = np.sum((p == 0) * (p != t)),np.sum((p == 0) * (p == t))
		
		LA_FIBR[n,0] = 2 * np.sum(p * t) / (np.sum(p) + np.sum(t))	# Dice score
		LA_FIBR[n,1] = tp / (tp + fn)								# Sensitivity
		LA_FIBR[n,2] = tn / (tn + fp)								# Specificity
		LA_FIBR[n,3] = np.sum(p * t) / np.sum((p + t) > 0)			# Intersection over union

		# surface to surface distance: mean (average symetric surface2surface dist) max (symetric Hausdorff dist) 
		LA_FIBR[n,4] = surfd(p,t,[0.625,0.625,2.5],1).mean()
		
		# SNR
		LA_FIBR[n,5] = np.abs(np.mean(lgemri[true == 3]) - np.mean(lgemri[true == 0])) / np.std(lgemri[true == 0])
		
		# fibrosis percentage
		LA_FIBR[n,6] = np.sum(t)/np.sum(true_LA_wall)
		LA_FIBR[n,7] = np.sum(p)/np.sum(pred_LA_wall)
		LA_FIBR[n,8] = np.abs(LA_FIBR[n,6] - LA_FIBR[n,7])*100
		
		# 2D Kolmogorov-Smirnov statistic
		t_com,p_com             = ndimage.measurements.center_of_mass(true == 3),ndimage.measurements.center_of_mass(pred == 3)
		t_pts,p_pts             = np.where(t),np.where(p)
		t_com,p_com,t_pts,p_pts = np.array(t_com),np.array(p_com),np.array(t_pts).T,np.array(p_pts).T
		t_pts,p_pts             = t_pts-t_com, p_pts-p_com
		
		KS_x,KS_y,KS_z = ks_2samp(t_pts[:,0],p_pts[:,0]),ks_2samp(t_pts[:,1],p_pts[:,1]),ks_2samp(t_pts[:,2],p_pts[:,2])
		LA_FIBR[n,9]   = 1 - np.mean([KS_x[0],KS_y[0],KS_z[0]])
		LA_FIBR[n,10]  = np.min([KS_x[1],KS_y[1],KS_z[1]])
		
		# correlation coefficients
		#LA_FIBR[n,11] = pearsonr(t,p)[0]

# Save data to csv output
if True: # True False
	pd.DataFrame(RA_FIBR).to_csv("RAfibrosis.csv",header=column_headers,index=False)
	pd.DataFrame(LA_FIBR).to_csv("LAfibrosis.csv",header=column_headers,index=False)

# summarize
print("\n\n---------------------- RA Fibrosis")
print("Dice Score: "+str(np.mean(RA_FIBR[:,0])))
print("S2S Distance: "+str(np.mean(RA_FIBR[:,4])))
print("Average Fibrosis % Difference: "+str(np.mean(RA_FIBR[:,8])))
print("PCC: "+str(np.mean(RA_FIBR[:,11])))

print("\n\n---------------------- LA Fibrosis")
print("Dice Score: "+str(np.mean(LA_FIBR[:,0])))
print("S2S Distance: "+str(np.mean(LA_FIBR[:,4])))
print("Average Fibrosis % Difference: "+str(np.mean(LA_FIBR[:,8])))
print("PCC: "+str(np.mean(LA_FIBR[:,11])))
