import sys
import os
import numpy as np
import cv2
import SimpleITK as sitk
import h5py
from scipy import ndimage
from QingXia_Utils import equalize_adapthist_3d
from scipy.ndimage.filters import uniform_filter
from scipy.interpolate import RegularGridInterpolator

### Helper functions
def load_nrrd(full_path_filename):
	# this function loads .nrrd files into a 3D matrix and outputs it
	# the input is the specified file path
	# the output is the N x A x B for N slices of sized A x B
	# after rolling, the output is the A x B x N
	data = sitk.ReadImage(full_path_filename)						# read in image
	data = sitk.Cast(sitk.RescaleIntensity(data),sitk.sitkUInt8)	# convert to 8 bit (0-255)
	data = sitk.GetArrayFromImage(data)								# convert to numpy array
	data = np.rollaxis(data,0,3)
	return(data)

def create_folder(full_path_filename):
	# this function creates a folder if its not already existed
	if not os.path.exists(full_path_filename):
		os.makedirs(full_path_filename)
	return

# path size
n1 = 272 # x
n2 = 272 # y

train_Image,train_Label = [],[]
test_Image,test_Label,Barycenter = [],[],[]

### Waikato Initialization -------------------------------------------------------------------------------------------------------------------------------------------
N_train_patients = [0,6,7,10] # patients to use for train set
N_test_patients  = [1,2,3,4,5,8,9] # patients to use for test set

os.chdir("C:/Users/Administrator/Desktop/2020_Waikato")

# list all the files in training and testing sets
files = os.listdir()
files.remove("2020_Waikato_PointCloud")
files.remove("seg info.xlsx")

# sort files
files = np.array(files)
files = files[np.argsort([int(f.replace("v2","")) for f in files])]

### Waikato Data ----------------------------------------------------------------------------------------------------------------------------------------------
# Train: loop through all training patients
for i in N_train_patients:

	print(str(i+1)+" Processing Train Set: "+files[i])
	
	# list all files in lgemri and labels
	img_files,lab_files = os.listdir(files[i]+"/lgemri"),os.listdir(files[i]+"/label")
	
	# load data, all must be 640 x 640 x 44
	temp_img,temp_lab = np.zeros([640,640,44]),np.zeros([640,640,44])
	for n in range(temp_img.shape[2]):
	
		# load image 8-bit
		temp_img[:,:,n] = cv2.imread(os.path.join(files[i]+"/lgemri",img_files[n]),cv2.IMREAD_GRAYSCALE)
		
		# load label (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo, 5 = septum)
		temp_lab[:,:,n] = cv2.imread(os.path.join(files[i]+"/label",lab_files[n]),cv2.IMREAD_GRAYSCALE)

	# normalize data
	temp_img = equalize_adapthist_3d(temp_img / np.max(temp_img))
	
	# filter
	temp_lab[temp_lab == 1] = 2 # RA wall set as same label as LA wall
	temp_lab[temp_lab == 0] = 1 # make background 1
	temp_lab -= 1               # subtract 1 so that background is set back to 0

	for n in range(len(lab_files)):
	
		# if there are positive pixels in the slice
		if np.sum(temp_lab[:,:,n]) > 0:
			
			# find the center of mass of the mask
			midpoint = ndimage.measurements.center_of_mass(temp_lab[:,:,n] > 0)

			# extract the patches from the midpoint
			n11,n12 = int(midpoint[0] - n1//2),int(midpoint[0] + n1//2)
			n21,n22 = int(midpoint[1] - n2//2),int(midpoint[1] + n2//2)

			# local image label for scan
			train_Image.append(temp_img[n11:n12,n21:n22,n])
			train_Label.append(temp_lab[n11:n12,n21:n22,n])

# Test: loop through all training patients
for i in N_test_patients:

	print(str(i+1)+" Processing Test Set: "+files[i])
	
	# list all files in lgemri and labels
	img_files,lab_files = os.listdir(files[i]+"/lgemri"),os.listdir(files[i]+"/label")

	# load data, all must be 640 x 640 x 44
	temp_img,temp_lab,temp_mid = np.zeros([576,576,44]),np.zeros([576,576,44]),np.zeros([44,2])
	for n in range(temp_img.shape[2]):
	
		# load image 8-bit
		temp1 = cv2.imread(os.path.join(files[i]+"/lgemri",img_files[n]),cv2.IMREAD_GRAYSCALE)
		
		# load label (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo, 5 = septum)
		temp2 = cv2.imread(os.path.join(files[i]+"/label",lab_files[n]),cv2.IMREAD_GRAYSCALE)
		
		# crop to utah size
		x,y = temp1.shape
		temp_img[:,:,n] = temp1[(x//2-288):(x//2+288),(y//2-288):(y//2+288)]
		temp_lab[:,:,n] = temp2[(x//2-288):(x//2+288),(y//2-288):(y//2+288)]
		
		# find the center of mass of the mask
		temp_mid[n,:] = ndimage.measurements.center_of_mass(temp_lab[:,:,n] > 0)

	# normalize data
	temp_img = equalize_adapthist_3d(temp_img / np.max(temp_img))
	
	# filter
	temp_lab[temp_lab == 1] = 2 # RA wall set as same label as LA wall
	temp_lab[temp_lab == 0] = 1 # make background 1
	temp_lab -= 1               # subtract 1 so that background is set back to 0

	# local image label for scan
	test_Image.append(temp_img)
	test_Label.append(temp_lab)
	Barycenter.append(temp_mid)

### Utah Initialization -------------------------------------------------------------------------------------------------------------------------------------------
N_train_patients = 27	# number of patients to use from train set

os.chdir("C:/Users/Administrator/Desktop/Utah Bi-Atria")

# list all the files in training and testing sets
files = os.listdir()

### Utah Data ----------------------------------------------------------------------------------------------------------------------------------------------
# Train Data: loop through all training patients
for i in range(N_train_patients): # [5,6,7]: #

	print(str(i+1)+" Processing Train Set: "+files[i])
	
	pat_files = os.listdir(files[i])
	
	for j in range(len(pat_files)):
		
		# read in the MRI scan and contrast normalization
		patient_3DMRI_scan = load_nrrd(os.path.join(files[i],pat_files[j],'lgemri.nrrd'))
		patient_3DMRI_scan = equalize_adapthist_3d(patient_3DMRI_scan / np.max(patient_3DMRI_scan))
		
		# load LA endo (with PVs) 
		laendo = load_nrrd(os.path.join(files[i],pat_files[j],'laendo.nrrd'))//255
		
		# cavity labels
		lab_folder = os.path.join(files[i],pat_files[j],"CARMA_"+files[i][5:]+"_"+pat_files[j]+"_full")
		lab_files  = os.listdir(lab_folder)
		
		for n in range(len(lab_files)):
			
			# load label (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo, 5 = septum)
			temp_lab = cv2.imread(os.path.join(lab_folder,lab_files[n]),cv2.IMREAD_GRAYSCALE)
			
			# add laendo
			temp_lab[laendo[:,:,n]==1] = 4
			
			# filter
			temp_lab[temp_lab > 4]  = 0 # remove septum
			temp_lab[temp_lab == 1] = 2 # RA wall set as same label as LA wall
			temp_lab[temp_lab == 0] = 1 # make background 1
			temp_lab -= 1               # subtract 1 so that background is set back to 0
			
			# if there are positive pixels in the slice
			if np.sum(temp_lab) > 0:
				
				# find the center of mass of the mask
				midpoint = ndimage.measurements.center_of_mass(temp_lab > 0)
			
				# extract the patches from the midpoint
				n11,n12 = int(midpoint[0] - n1//2),int(midpoint[0] + n1//2)
				n21,n22 = int(midpoint[1] - n2//2),int(midpoint[1] + n2//2)

				# local image label for scan
				train_Image.append(patient_3DMRI_scan[n11:n12,n21:n22,n])
				train_Label.append(temp_lab[n11:n12,n21:n22])

# Test Data: loop through all training patients
for i in range(N_train_patients,len(files)):

	print(str(i+1)+" Processing Test Set: "+files[i])
	
	pat_files = os.listdir(files[i])
	
	for j in range(len(pat_files)):
		
		# read in the MRI scan and contrast normalization
		patient_3DMRI_scan = load_nrrd(os.path.join(files[i],pat_files[j],'lgemri.nrrd'))
		patient_3DMRI_scan = equalize_adapthist_3d(patient_3DMRI_scan / np.max(patient_3DMRI_scan))
		
		# crop shape so all outputs are 576 x 576
		x,y,z = patient_3DMRI_scan.shape
		patient_3DMRI_scan = patient_3DMRI_scan[(x//2-288):(x//2+288),(y//2-288):(y//2+288),:]
		
		# load LA endo (with PVs) 
		laendo = load_nrrd(os.path.join(files[i],pat_files[j],'laendo.nrrd'))//255
		
		# cavity labels
		lab_folder = os.path.join(files[i],pat_files[j],"CARMA_"+files[i][5:]+"_"+pat_files[j]+"_full")
		lab_files  = os.listdir(lab_folder)

		# loop through all the slices
		patient_lab,patient_mid = np.zeros([576,576,44]),np.zeros([44,2])

		for n in range(len(lab_files)):
			
			# load label (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo, 5 = septum)
			temp_lab = cv2.imread(os.path.join(lab_folder,lab_files[n]),cv2.IMREAD_GRAYSCALE)
			
			# add laendo
			temp_lab[laendo[:,:,n]==1] = 4
			
			# filter 
			temp_lab[temp_lab > 4]  = 0 # remove septum
			temp_lab[temp_lab == 1] = 2 # RA wall set as same label as LA wall
			temp_lab[temp_lab == 0] = 1 # make background 1
			temp_lab -= 1               # subtract 1 so that background is set back to 0
			
			# crop so all outputs are 576 x 576
			temp_lab = temp_lab[(x//2-288):(x//2+288),(y//2-288):(y//2+288)]
			
			# find the center of mass of the mask
			patient_lab[:,:,n] = temp_lab
			patient_mid[n,:]   = ndimage.measurements.center_of_mass(temp_lab > 0)

		# local image label for scan
		test_Image.append(patient_3DMRI_scan)
		test_Label.append(patient_lab)
		Barycenter.append(patient_mid)
	
### Save Data ----------------------------------------------------------------------------------------------------------------------------------------------
os.chdir("C:/Users/Administrator/Desktop")

create_folder("UtahWaikato Test Set")
create_folder("UtahWaikato Test Set/log")
create_folder("UtahWaikato Test Set/Prediction Sample")

# Train Data
train_Image,train_Label = np.array(train_Image),np.array(train_Label)

# encoding label to neural network output format (0 = background, 1 = RA+LA wall, 2 = RA endo, 3 = LA endo)
temp = np.empty(shape=[train_Label.shape[0],n1,n2,4])
for i in range(4):
	x = train_Label == i
	temp[:,:,:,i] = x

train_Image,train_Label = np.reshape(train_Image,newshape=[-1,n1,n2,1]),np.reshape(temp,newshape=[-1,n1,n2,4])

# Test Data
test_Image,test_Label,Barycenter = np.array(test_Image),np.array(test_Label),np.array(Barycenter)

# create a HDF5 dataset
h5f = h5py.File('UtahWaikato Test Set/Training.h5','w')
h5f.create_dataset("image",data=train_Image)
h5f.create_dataset("label",data=train_Label)
h5f.close()

# create a HDF5 dataset
h5f = h5py.File('UtahWaikato Test Set/Testing.h5','w')
h5f.create_dataset("image",     data=test_Image)
h5f.create_dataset("label",     data=test_Label)
h5f.create_dataset("centroid",  data=Barycenter)
h5f.close()
