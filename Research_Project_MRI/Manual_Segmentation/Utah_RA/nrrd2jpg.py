# this script converts the nrrd images to tiff stacks

import sys
import os
import numpy as np
import cv2
import SimpleITK as sitk

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

create_folder("CavityMask")
create_folder("LAendoNoVeins")
create_folder("LAwall")
create_folder("LGEMRI")
create_folder("SeptumConnection")

laendo = load_nrrd("laendo_no_veins.nrrd")//255
lawall = load_nrrd("lawall.nrrd")//255
lgemri = load_nrrd("lgemri.nrrd")

#assert(np.all(laendo.shape == lawall.shape))
#assert(np.all(laendo.shape == lgemri.shape))
#assert(laendo.shape[2] == 44)

for i in range(lgemri.shape[2]):
        output_filename = "{0:03}".format(i+1)+".tif"
        cv2.imwrite("LAendoNoVeins/laendo"+output_filename,laendo[:,:,i])
        cv2.imwrite("LAwall/lawall"+output_filename,lawall[:,:,i])
        cv2.imwrite("LGEMRI/lgemri"+output_filename,lgemri[:,:,i])
