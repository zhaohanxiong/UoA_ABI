import os
import cv2
import copy
import scipy.io
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.filters import uniform_filter
from scipy.interpolate import RegularGridInterpolator

### --- Helper functions ---------------------------------------------------------------------------------------
def load_nrrd(full_path_filename):
	
	# this function loads .nrrd files into a 3D matrix and outputs it
	# the input is the specified file path
	# the output is the N x A x B for N slices of sized A x B
	
	data = sitk.ReadImage( full_path_filename )							# read in image
	data = sitk.Cast( sitk.RescaleIntensity(data), sitk.sitkUInt8 )		# convert to 8 bit (0-255)
	data = sitk.GetArrayFromImage( data )								# convert to numpy array
	
	return(data)

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

### --- Process All Patients ---------------------------------------------------------------------------------------

Image_Path = "C:/Users/zxio506/Desktop/Atria_Data/Utah Bi-Atria"
AWT_Path   = "C:/Users/zxio506/Desktop/Atria_Data/AWT Utah Bi-Atria"

pat_ID_files = os.listdir(Image_Path)

for pat_ID in pat_ID_files:

	status_files = os.listdir(os.path.join(Image_Path,pat_ID))
	
	for status in status_files:
		
		print(pat_ID + " " + status)
		
		# load image data
		lgemri = load_nrrd(os.path.join(Image_Path,pat_ID,status,"lgemri.nrrd"))
		lgemri = np.rollaxis(lgemri,0,3)
		
		laendo = load_nrrd(os.path.join(Image_Path,pat_ID,status,"laendo.nrrd"))//255
		laendo = np.rollaxis(laendo,0,3)

		# load manual segmentation data
		lab_loc   = os.listdir(os.path.join(Image_Path,pat_ID,status))
		lab_loc   = [s for s in lab_loc if "CARMA" in s][0]
		lab_files = os.listdir(os.path.join(Image_Path,pat_ID,status,lab_loc))
		label     = cv2.imread(os.path.join(Image_Path,pat_ID,status,lab_loc,lab_files[0]),cv2.IMREAD_GRAYSCALE)
		label     = np.zeros([label.shape[0],label.shape[1],len(lab_files)])

		for n in range(len(lab_files)):
			label[:,:,n] = cv2.imread(os.path.join(Image_Path,pat_ID,status,lab_loc,lab_files[n]),cv2.IMREAD_GRAYSCALE)

		label = label.astype(np.uint8)
		label[label>4] = 0

		# interpolation
		lgemri = interpolate_data_z(lgemri,2)

		RAwall = (label == 1).astype(np.uint8)
		LAwall = (label == 2).astype(np.uint8)
		RAcavi = (label == 3).astype(np.uint8)
		LAcavi = (label == 4).astype(np.uint8)

		interp_RAcavi = smooth3D_interpolate(RAcavi,20,2)
		interp_LAcavi = smooth3D_interpolate(LAcavi,20,2)

		interp_RAwall = smooth3D_interpolate(RAwall+RAcavi,20,2) - interp_RAcavi
		interp_RAwall[interp_RAwall<0] = 0
		interp_RAwall[interp_RAwall>1] = 0
		interp_LAwall = smooth3D_interpolate(LAwall+LAcavi,20,2) - interp_LAcavi
		interp_LAwall[interp_LAwall<0] = 0
		interp_LAwall[interp_LAwall>1] = 0

		# compile label into one variable
		interp_label = np.zeros([label.shape[0],label.shape[1],label.shape[2]*2])
		interp_label[interp_RAwall == 1] = 1
		interp_label[interp_LAwall == 1] = 2
		interp_label[interp_RAcavi == 1] = 3
		interp_label[interp_LAcavi == 1] = 4

		# output
		create_folder(os.path.join(AWT_Path,pat_ID+"_"+status))
		#create_folder(os.path.join(AWT_Path,pat_ID+"_"+status,"AWT"))
		create_folder(os.path.join(AWT_Path,pat_ID+"_"+status,"label"))
		create_folder(os.path.join(AWT_Path,pat_ID+"_"+status,"lgemri"))

		for i in range(lgemri.shape[2]):
			
			img_name = "{0:03}".format(i+1)+".tif"
			cv2.imwrite(os.path.join(AWT_Path,pat_ID+"_"+status,"lgemri","lgemri"+img_name),lgemri[:,:,i])
			cv2.imwrite(os.path.join(AWT_Path,pat_ID+"_"+status,"label","label"+img_name),interp_label[:,:,i])
