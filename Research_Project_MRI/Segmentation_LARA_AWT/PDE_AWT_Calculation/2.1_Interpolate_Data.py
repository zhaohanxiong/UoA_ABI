import os
import cv2
import copy
import scipy.io
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter
from scipy.interpolate import RegularGridInterpolator

####################################
pat_ID        = "00007v2"
interp_factor = 4
####################################

### --- Helper functions ---------------------------------------------------------------------------------------
def create_folder(full_path_filename):
	
	# this function creates a folder if its not already existed
	if not os.path.exists(full_path_filename):
		os.makedirs(full_path_filename)

	return
	

def interpolate_data_z(ImageIn,factor):
	
	# this function interpolates the 3D data in the z direction depending on the factor
	Nx,Ny,Nz = ImageIn.shape
	x,y,z = np.linspace(1,Nx,Nx),np.linspace(1,Ny,Ny),np.linspace(1,Nz,Nz)
	interp_func = RegularGridInterpolator((x,y,z),ImageIn,method="linear")
	[xi,yi,zi] = np.meshgrid(x,y,np.linspace(1,Nz,factor*Nz),indexing='ij')
	ImageIn = interp_func( np.stack([xi,yi,zi],axis=3) )
	
	return(ImageIn)

def smooth3D_interpolate(data,threshold=20,factor=2):
	
	# this function interpolates the MRI and smoothes it in 3D
	data[data>=1] = 1
	data[data!=1] = 0
	data[data==1] = 50
	data = interpolate_data_z(data,factor)
	data = uniform_filter(data,5)
	data[data <  threshold] = 0
	data[data >= threshold] = 50
	data = data//50
	
	return(data)

### --- Process 1 Patient ---------------------------------------------------------------------------------------

# load image data
img_files = os.listdir(os.path.join(pat_ID,"lgemri"))
lgemri    = cv2.imread(os.path.join(pat_ID,"lgemri",img_files[0]),cv2.IMREAD_GRAYSCALE)
lgemri    = np.zeros([lgemri.shape[0],lgemri.shape[1],len(img_files)])

# load manual segmentation data
lab_files = os.listdir(os.path.join(pat_ID,"AWT"))
label     = cv2.imread(os.path.join(pat_ID,"AWT",lab_files[0]),cv2.IMREAD_GRAYSCALE)
label     = np.zeros([label.shape[0],label.shape[1],len(lab_files)])

for n in range(len(img_files)):
	lgemri[:,:,n] = cv2.imread(os.path.join(pat_ID,"lgemri",img_files[n]),cv2.IMREAD_GRAYSCALE)
	label[:,:,n]  = cv2.imread(os.path.join(pat_ID,"AWT",lab_files[n]),cv2.IMREAD_GRAYSCALE)

lgemri = lgemri.astype(np.uint8)
label  = label.astype(np.uint8)
label[label>4] = 0

# interpolation lgemri
lgemri = interpolate_data_z(lgemri, interp_factor)

# interpolation label
RAwall = (label == 1).astype(np.uint8)
LAwall = (label == 2).astype(np.uint8)
RAcavi = (label == 3).astype(np.uint8)
LAcavi = (label == 4).astype(np.uint8)

interp_RAcavi = smooth3D_interpolate(RAcavi,20, interp_factor)
interp_LAcavi = smooth3D_interpolate(LAcavi,20, interp_factor)

interp_RAwall = smooth3D_interpolate(RAwall+RAcavi,20, interp_factor) - interp_RAcavi
interp_RAwall[interp_RAwall<0] = 0
interp_RAwall[interp_RAwall>1] = 0
interp_LAwall = smooth3D_interpolate(LAwall+LAcavi,20, interp_factor) - interp_LAcavi
interp_LAwall[interp_LAwall<0] = 0
interp_LAwall[interp_LAwall>1] = 0

# compile label into one variable
interp_label = np.zeros([label.shape[0],label.shape[1],label.shape[2] * interp_factor])
interp_label[interp_RAwall == 1] = 1
interp_label[interp_LAwall == 1] = 2
interp_label[interp_RAcavi == 1] = 3
interp_label[interp_LAcavi == 1] = 4

# output
create_folder(os.path.join(pat_ID,"interp_lgemri"))
create_folder(os.path.join(pat_ID,"interp_AWT"))

for i in range(lgemri.shape[2]):
	
	img_name = "{0:03}".format(i+1)+".tif"
	cv2.imwrite(os.path.join(pat_ID,"interp_lgemri","lgemri"+img_name),lgemri[:,:,i])
	cv2.imwrite(os.path.join(pat_ID,"interp_AWT","awt"+img_name),interp_label[:,:,i])
