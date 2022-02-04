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

# sphericity of a 3D volume
def sphericity(x):

	# volume of object
	Vp = np.sum(x)*0.625*0.625*2.5
	
	# surface area of object
	Ap = (np.sum(x) - np.sum(morphology.binary_erosion(x, iterations=1)))*0.625
	
	# formula for sphericity
	thi = np.pi**(1/3) * (6*Vp)**(2/3) / Ap
	
	return thi * 100

# set directory location
os.chdir("C:/Users/zxio506/Desktop/UtahWaikato Test Set")

# list files
files = os.listdir("Prediction Sample")

# make files
if False:
	os.mkdir("Image Stack")

# store output scores
column_headers = ["DSC","sty","spy","IoU",
				  "pred_dia","true_dia","dia_err","dia_per",
				  "pred_vol","true_vol","vol_err","vol_per", 
				  "s2s_dist","SNR",
				  "pred_sphericity","true_sphericity","sphe_err","sphe_per"]

test_patients  = os.listdir("Prediction Sample")

LARAepi_score = np.zeros([len(test_patients),len(column_headers)])
RAendo_score  = np.zeros([len(test_patients),len(column_headers)])
LAendo_score  = np.zeros([len(test_patients),len(column_headers)])

# loop through all files
for n in range(len(files)):
	
	print("Processing "+files[n])
	
	# read data
	dat = scipy.io.loadmat("Prediction Sample/"+files[n])
	true,pred = dat["true"].astype(np.uint8),dat["pred"].astype(np.uint8)
	
	# post-process prediction
	epi    = pred >= 1
	RAendo = pred == 2
	LAendo = pred == 3
	
	# # structure useful for 3D processing
	# cross3D = np.array([[[0,0,0],[0,1,0],[0,0,0]],
						# [[0,1,0],[1,1,1],[0,1,0]],
						# [[0,0,0],[0,1,0],[0,0,0]]])
	
	# # fill holes in every axis, then in 3D
	# for i in range(pred.shape[0]):	
		# epi[i,:,:]	  = morphology.binary_fill_holes(epi[i,:,:], 	structure=np.ones((3,3)))
		# RAendo[i,:,:] = morphology.binary_fill_holes(RAendo[i,:,:], structure=np.ones((3,3)))
		# LAendo[i,:,:] = morphology.binary_fill_holes(LAendo[i,:,:], structure=np.ones((3,3)))
	# for i in range(pred.shape[1]):	
		# epi[:,i,:]	  = morphology.binary_fill_holes(epi[:,i,:], 	structure=np.ones((3,3)))
		# RAendo[:,i,:] = morphology.binary_fill_holes(RAendo[:,i,:], structure=np.ones((3,3)))
		# LAendo[:,i,:] = morphology.binary_fill_holes(LAendo[:,i,:], structure=np.ones((3,3)))
	# for i in range(pred.shape[2]):	
		# epi[:,:,i]	  = morphology.binary_fill_holes(epi[:,:,i], 	structure=np.ones((3,3)))
		# RAendo[:,:,i] = morphology.binary_fill_holes(RAendo[:,:,i], structure=np.ones((3,3)))
		# LAendo[:,:,i] = morphology.binary_fill_holes(LAendo[:,:,i], structure=np.ones((3,3)))

	# # remove smallest islands
	# mask_reg, mask_lab  = label(epi, structure = cross3D)
	# reg_size 			= np.array([np.sum(mask_reg[mask_reg == i]) for i in np.unique(mask_reg)])
	# epi 				= mask_reg == np.argmax(reg_size)
	
	# mask_reg, mask_lab  = label(RAendo, structure = cross3D)
	# reg_size 			= np.array([np.sum(mask_reg[mask_reg == i]) for i in np.unique(mask_reg)])
	# RAendo 				= mask_reg == np.argmax(reg_size)
	
	# mask_reg, mask_lab  = label(LAendo, structure = cross3D)
	# reg_size 			= np.array([np.sum(mask_reg[mask_reg == i]) for i in np.unique(mask_reg)])
	# LAendo 				= mask_reg == np.argmax(reg_size)

	# recombine into one label
	pred = np.zeros_like(pred)
	pred[epi==1] 	= 1
	pred[RAendo==1] = 2
	pred[LAendo==1] = 3
	
	# process lgemri data
	lgemri = dat["lgemri"]
	lgemri = lgemri/np.max(lgemri)*255
	lgemri = lgemri.astype(np.uint8)
	
	if False:
	
		# write to image stack
		os.mkdir("Image Stack/"+test_patients[n])
		os.mkdir("Image Stack/"+test_patients[n]+"/true")
		os.mkdir("Image Stack/"+test_patients[n]+"/pred")
		os.mkdir("Image Stack/"+test_patients[n]+"/lgemri")
		
		for i in range(true.shape[2]):
			img_name = "{0:03}".format(i+1)+".tif"
			cv2.imwrite("Image Stack/"+test_patients[n]+"/true/"+img_name,true[:,:,i])
			cv2.imwrite("Image Stack/"+test_patients[n]+"/pred/"+img_name,pred[:,:,i])
			cv2.imwrite("Image Stack/"+test_patients[n]+"/lgemri/"+img_name,lgemri[:,:,i])
	
	if True:
		## ----------------------------------------------------------------------------------------------- LARA EPI
		t,p = true>=1,pred>=1

		tp,fp = np.sum((p == 1) * (p == t)),np.sum((p == 1) * (p != t))
		fn,tn = np.sum((p == 0) * (p != t)),np.sum((p == 0) * (p == t))
		
		LARAepi_score[n,0] = 2 * np.sum(p * t) / (np.sum(p) + np.sum(t))	# Dice score
		LARAepi_score[n,1] = tp / (tp + fn)								# Sensitivity
		LARAepi_score[n,2] = tn / (tn + fp)								# Specificity
		LARAepi_score[n,3] = np.sum(p * t) / np.sum((p + t) > 0)			# Intersection over union

		# LA diameter
		p_dia,t_dia  = np.max(np.sum(p,0))*0.625,np.max(np.sum(t,0))*0.625
		LARAepi_score[n,4] = p_dia										# predicted diameter
		LARAepi_score[n,5] = t_dia										# true diameter
		LARAepi_score[n,6] = np.abs(t_dia - p_dia)						# diameter error in pixels
		LARAepi_score[n,7] = np.abs(t_dia - p_dia) / t_dia				# diameter error in percentage		
		
		# LA volume
		p_vol,t_vol  = np.sum(p)*0.625*0.625*2.5,np.sum(t)*0.625*0.625*2.5
		LARAepi_score[n,8]  = p_vol										# predicted volume
		LARAepi_score[n,9]  = t_vol										# true volume
		LARAepi_score[n,10] = np.abs(t_vol - p_vol)						# volume error in pixels
		LARAepi_score[n,11] = np.abs(t_vol - p_vol) / t_vol				# volume error in percentage

		# surface to surface distance: mean (average symetric surface2surface dist) max (symetric Hausdorff dist) 
		LARAepi_score[n,12] = surfd(p,t,[0.625,0.625,2.5],1).mean()
		
		# SNR
		LARAepi_score[n,13] = np.abs(np.mean(lgemri[true >= 1]) - np.mean(lgemri[true == 0])) / np.std(lgemri[true == 0])
		
		# sphericity
		p_sphe,t_sphe = sphericity(p),sphericity(t)
		LARAepi_score[n,14] = p_sphe
		LARAepi_score[n,15] = t_sphe
		LARAepi_score[n,16] = np.abs(t_sphe - p_sphe)
		LARAepi_score[n,17] = np.abs(t_sphe - p_sphe) / t_sphe
		
		## ----------------------------------------------------------------------------------------------- RA ENDO
		t,p = true==2,pred==2

		tp,fp = np.sum((p == 1) * (p == t)),np.sum((p == 1) * (p != t))
		fn,tn = np.sum((p == 0) * (p != t)),np.sum((p == 0) * (p == t))
		
		RAendo_score[n,0] = 2 * np.sum(p * t) / (np.sum(p) + np.sum(t))	# Dice score
		RAendo_score[n,1] = tp / (tp + fn)								# Sensitivity
		RAendo_score[n,2] = tn / (tn + fp)								# Specificity
		RAendo_score[n,3] = np.sum(p * t) / np.sum((p + t) > 0)			# Intersection over union

		# RA diameter
		p_dia,t_dia  = np.max(np.sum(p,1))*0.625,np.max(np.sum(t,1))*0.625
		RAendo_score[n,4] = p_dia											# predicted diameter
		RAendo_score[n,5] = t_dia											# true diameter
		RAendo_score[n,6] = np.abs(t_dia - p_dia)							# diameter error in pixels
		RAendo_score[n,7] = np.abs(t_dia - p_dia) / t_dia					# diameter error in percentage		
		
		# RA volume
		p_vol,t_vol  = np.sum(p)*0.625*0.625*2.5,np.sum(t)*0.625*0.625*2.5
		RAendo_score[n,8]  = p_vol										# predicted volume
		RAendo_score[n,9]  = t_vol										# true volume
		RAendo_score[n,10] = np.abs(t_vol - p_vol)						# volume error in pixels
		RAendo_score[n,11] = np.abs(t_vol - p_vol) / t_vol				# volume error in percentage

		# surface to surface distance: mean (average symetric surface2surface dist) max (symetric Hausdorff dist) 
		RAendo_score[n,12] = surfd(p,t,[0.625,0.625,2.5],1).mean()
		
		# SNR
		RAendo_score[n,13] = np.abs(np.mean(lgemri[true == 2]) - np.mean(lgemri[true == 0])) / np.std(lgemri[true == 0])
		
		# sphericity
		p_sphe,t_sphe = sphericity(p),sphericity(t)
		RAendo_score[n,14] = p_sphe
		RAendo_score[n,15] = t_sphe
		RAendo_score[n,16] = np.abs(t_sphe - p_sphe)
		RAendo_score[n,17] = np.abs(t_sphe - p_sphe) / t_sphe

		## ----------------------------------------------------------------------------------------------- LA ENDO
		t,p = true==3,pred==3

		tp,fp = np.sum((p == 1) * (p == t)),np.sum((p == 1) * (p != t))
		fn,tn = np.sum((p == 0) * (p != t)),np.sum((p == 0) * (p == t))
		
		LAendo_score[n,0] = 2 * np.sum(p * t) / (np.sum(p) + np.sum(t))	# Dice score
		LAendo_score[n,1] = tp / (tp + fn)								# Sensitivity
		LAendo_score[n,2] = tn / (tn + fp)								# Specificity
		LAendo_score[n,3] = np.sum(p * t) / np.sum((p + t) > 0)			# Intersection over union

		# LA diameter
		p_dia,t_dia  = np.max(np.sum(p,0))*0.625,np.max(np.sum(t,0))*0.625
		LAendo_score[n,4] = p_dia											# predicted diameter
		LAendo_score[n,5] = t_dia											# true diameter
		LAendo_score[n,6] = np.abs(t_dia - p_dia)							# diameter error in mm
		LAendo_score[n,7] = np.abs(t_dia - p_dia) / t_dia					# diameter error in percentage		
		
		# LA volume
		p_vol,t_vol  = np.sum(p)*0.625*0.625*2.5,np.sum(t)*0.625*0.625*2.5
		LAendo_score[n,8]  = p_vol										# predicted volume
		LAendo_score[n,9]  = t_vol										# true volume
		LAendo_score[n,10] = np.abs(t_vol - p_vol)						# volume error in mm
		LAendo_score[n,11] = np.abs(t_vol - p_vol) / t_vol				# volume error in percentage

		# surface to surface distance: mean (average symetric surface2surface dist) max (symetric Hausdorff dist) 
		LAendo_score[n,12] = surfd(p,t,[0.625,0.625,2.5],1).mean()
		
		# SNR
		LAendo_score[n,13] = np.abs(np.mean(lgemri[true == 3]) - np.mean(lgemri[true == 0])) / np.std(lgemri[true == 0])
		
		# sphericity
		p_sphe,t_sphe = sphericity(p),sphericity(t)
		LAendo_score[n,14] = p_sphe
		LAendo_score[n,15] = t_sphe
		LAendo_score[n,16] = np.abs(t_sphe - p_sphe)
		LAendo_score[n,17] = np.abs(t_sphe - p_sphe) / t_sphe

# Save data to csv output
pd.DataFrame(LARAepi_score).to_csv("LARAepi.csv",header=column_headers,index=False)
pd.DataFrame(RAendo_score).to_csv("RAendo.csv",header=column_headers,index=False)
pd.DataFrame(LAendo_score).to_csv("LAendo.csv",header=column_headers,index=False)

# summarize
print("\n\n---------------------- LA+RA Epi Evaluation")
print("Dice Score: "+str(np.mean(LARAepi_score[:,0])))
print("      S2SD: "+str(np.mean(LARAepi_score[:,12])))

print("\n\n---------------------- RA Endo Evaluation")
print("Dice Score: "+str(np.mean(RAendo_score[:,0])))
print("      S2SD: "+str(np.mean(RAendo_score[:,12])))

print("\n\n---------------------- LA Endo Evaluation")
print("Dice Score: "+str(np.mean(LAendo_score[:,0])))
print("      S2SD: "+str(np.mean(LAendo_score[:,12])))

#for i in SNR: print( i)
'''
figure
n = 25
subplot(1,2,1)
imagesc(true(:,:,n))
subplot(1,2,2)
imagesc(pred(:,:,n))
'''
