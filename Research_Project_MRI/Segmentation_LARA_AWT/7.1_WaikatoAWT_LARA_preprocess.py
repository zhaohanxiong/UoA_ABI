import os
import cv2
import sys
import h5py
import scipy.io
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from QingXia_Utils import equalize_adapthist_3d

def create_folder(full_path_filename):
	# this function creates a folder if its not already existed
	if not os.path.exists(full_path_filename):
		os.makedirs(full_path_filename)
	return

# path size
n1 = 272 # x
n2 = 272 # y

### Utah Data ----------------------------------------------------------------------------------------------------------------------------------------------
os.chdir("C:/Users/zxio506/Desktop/Atria_Data/AWT Waikato")

N_train_patients = [0,6,7,10] # patients to use for train set
N_test_patients  = [1,2,3,4,5,8,9] # patients to use for test set

# list all the files in training and testing sets
files = os.listdir()
files = [f for f in files if "0000" in f]

# Train Data: loop through all training patients
train_Image,train_Label,train_AWT = [],[],[]
test_Label,Barycenter,test_AWT = [],[],[]

for i in N_train_patients:

	print(str(i+1)+" Processing Train Set: "+files[i])

	# load awt file
	awt_file = scipy.io.loadmat(files[i]+"/AWT.mat")

	# get awt and labels
	temp_awt = awt_file["AWT"]
	temp_lab = awt_file["label"]
	temp_lab[temp_lab > 4]  = 0
	
	# load label data one slice at a time
	for n in range(temp_lab.shape[2]):

		# if there are positive pixels in the slice
		if np.sum(temp_lab[:,:,n]) > 0:
			
			# find the center of mass of the mask
			midpoint = ndimage.measurements.center_of_mass(temp_lab[:,:,n] > 0)
		
			# extract the patches from the midpoint
			n11,n12 = int(midpoint[0] - n1//2),int(midpoint[0] + n1//2)
			n21,n22 = int(midpoint[1] - n2//2),int(midpoint[1] + n2//2)

			# local image label for scan
			train_AWT.append(temp_awt[n11:n12,n21:n22,n])
			train_Label.append(temp_lab[n11:n12,n21:n22,n])

# Test Data: loop through all training patients
for i in N_test_patients:

	print(str(i+1)+" Processing Test Set: "+files[i])

	# load awt file
	awt_file = scipy.io.loadmat(files[i]+"/AWT.mat")

	# get awt and labels
	temp_awt = awt_file["AWT"]
	temp_lab = awt_file["label"]
	
	# load awt file
	x,y,z    = temp_awt.shape
	temp_awt = temp_awt[(x//2-288):(x//2+288),(y//2-288):(y//2+288),:]
	temp_lab = temp_lab[(x//2-288):(x//2+288),(y//2-288):(y//2+288),:]
	
	# loop through all the label slices
	patient_lab,patient_mid = np.zeros([576,576,176]),np.zeros([176,2])

	for n in range(temp_lab.shape[2]):
	
		# crop so all inputs are 576 x 576
		patient_lab[:,:,n] = temp_lab[:,:,n]
		patient_mid[n,:]   = ndimage.measurements.center_of_mass(temp_lab[:,:,n] > 0)

	# local image label for scan
	test_AWT.append(temp_awt)
	test_Label.append(patient_lab)
	Barycenter.append(patient_mid)
	
### Save Data ----------------------------------------------------------------------------------------------------------------------------------------------
os.chdir("C:/Users/zxio506/Desktop")

create_folder("WaikatoAWT Test Set")
create_folder("WaikatoAWT Test Set/log")
create_folder("WaikatoAWT Test Set/Prediction Sample")

# Train Data
train_AWT,train_Label = np.reshape(np.array(train_AWT),newshape=[-1,n1,n2,1]),np.reshape(np.array(train_Label),newshape=[-1,n1,n2,1])

# Test Data
test_Label,Barycenter,test_AWT = np.array(test_Label),np.array(Barycenter),np.array(test_AWT)

# create a HDF5 dataset
print("---------- Saving Training Data")
h5f = h5py.File('WaikatoAWT Test Set/Training.h5','w')
h5f.create_dataset("label", data=train_Label)
h5f.create_dataset("awt",   data=train_AWT)
h5f.close()

# create a HDF5 dataset
print("---------- Saving Testing Data")
h5f = h5py.File('WaikatoAWT Test Set/Testing.h5','w')
h5f.create_dataset("label",    data=test_Label)
h5f.create_dataset("centroid", data=Barycenter)
h5f.create_dataset("awt",      data=test_AWT)
h5f.close()
