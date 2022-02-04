import sys
import os
import numpy as np
import h5py
import scipy.io
import matplotlib.pyplot as plt
import cv2
import time
import copy
import random
from sklearn.cross_validation import train_test_split

os.chdir("C:\\Users\\Administrator\\Desktop")

# set up window size for labels
factor 			= 4 # how much Phi was downsampled
sub_region_size = 8 # in original 240 x 240 resolution
n_window 		= int(sub_region_size / factor)

# list files
files = os.listdir("ECP")
Phi_files = [s for s in files if s.startswith("Phi")]
PS_files  = [s for s in files if s.startswith("PS_trajectory")]

# initialization
test_files = np.array([1,5,8,10,11,16,24,26,28,47]) # which files to test
train,train_label,test,test_label,test_label_PS = [],[],[],[],[]

for n in range(len(Phi_files)):

	print("Processing file "+str(n+1)+": "+Phi_files[n])
	
	# check file names
	assert Phi_files[n].split("_")[-1] == PS_files[n].split("_")[-1], "\nFiles not the same!"
	
	# load files
	if n >= 24 and n <= 43:
		Phi = scipy.io.loadmat(os.path.join('ECP',Phi_files[n]))['Phi']
		PS  = scipy.io.loadmat(os.path.join('ECP',PS_files[n]))['PS_trajectory'][0::5,:]
	else:
		h5f = h5py.File(os.path.join('ECP',Phi_files[n]),'r')
		Phi = np.transpose(h5f['Phi'].value)
		h5f.close()
		PS  = scipy.io.loadmat(os.path.join('ECP',PS_files[n]))['PS_trajectory']
	
	# check  sample length
	assert Phi.shape[2] == PS.shape[0], "\nSample length not the same!"

	# find average centroid
	PS_mean = np.mean(np.array([np.mean(PS[i,0],axis=0) for i in range(len(PS)) if not np.isnan(np.mean(PS[i,0]))]),axis=0)
	
	# find interval where PS is valid
	PS_num = np.array([PS[i,0].shape[0] for i in range(len(PS))])
	PS_markers = np.where(np.array([np.all(PS_num[i:i+10]<=3) for i in range(len(PS_num)-10)]))
	PS_start,PS_end = np.min(PS_markers),np.max(PS_markers)

	if not np.any(n == test_files):
		
		# create training labels, co-ordinates normalized to 0-1
		for i in range(PS_start,PS_end):
			# extract point
			PS_main = PS[i,0][np.argmin(np.mean(np.abs(PS[i,0] - PS_mean),axis=1)),:]
			# append label and data
			if not np.any(np.isnan(PS_main)):
				train.append(Phi[:,:,i][:,:,None])
				train_label.append([PS_main[0]/Phi.shape[0],PS_main[1]/Phi.shape[1]])

	else:

		# create label
		label_map,label_temp2,test_label_temp = np.zeros([Phi.shape[0]//n_window,Phi.shape[1]//n_window]),[],[]
		for i in range(PS_start,PS_end):
			# extract point
			PS_main = PS[i,0][np.argmin(np.mean(np.abs(PS[i,0] - PS_mean),axis=1)),:]
			# append labels and data
			if not np.any(np.isnan(PS_main)):
				label_map[int(np.round(PS_main[0]//n_window)),int(np.round(PS_main[1]//n_window))] += 1
				label_temp2.append([PS_main[0]/Phi.shape[0],PS_main[1]/Phi.shape[1]])
				test_label_temp.append(Phi[:,:,i][:,:,None])
		
		#label_map[label_map < (np.max(label_map)*0.1)] = 0
		#label_map[label_map >=(np.max(label_map)*0.1)] = 1
		
		# append label
		test_label.append(label_map)
		test_label_PS.append(np.array(label_temp2))
		test.append(np.array(test_label_temp))

train,train_label,test,test_label,test_label_PS = np.array(train),np.array(train_label),np.array(test),np.array(test_label),np.array(test_label_PS)

# shuffle data
train,_,train_label,_ = train_test_split(train,train_label,test_size=0,random_state=42)

# normalize data
train_mean 	= np.mean(train)
train_sd 	= np.std(train)

train 	= (train - train_mean)/train_sd

# save to file
scipy.io.savemat("Electrogram_Data.mat",mdict={'train':train,'train_label':train_label,'test':test,'test_label':test_label,
                                               'test_label_PS':test_label_PS,'train_mean':train_mean,'train_sd':train_sd})