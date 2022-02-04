import scipy.io
import numpy as np
import os
import shutil
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes

os.chdir("C:/Users/zxio506/Desktop/Atria_Data/CatheterPath/2020_Melbourne_PointCloud")

# load data
melbourne_patients = ["Export_AT-08_08_2017-13-26-43",   # 1
                      "Export_PVI-11_27_2020-16-11-08",  # 2
					 ]

for n_read in range(len(melbourne_patients)):

	dat = scipy.io.loadmat(melbourne_patients[n_read]+"/CT_raster.mat")

	# load the coordinates representing the rasterized mesh
	mask_coords = dat['CT']
	mask_coords[:,0] -= np.min(mask_coords[:,0])
	mask_coords[:,1] -= np.min(mask_coords[:,1])
	mask_coords[:,2] -= np.min(mask_coords[:,2])
	
	# rescale to make it 200 x 200 x 44
	mask_coords[:,0] = mask_coords[:,0]/np.max(mask_coords[:,0])*100 + 270
	mask_coords[:,1] = mask_coords[:,1]/np.max(mask_coords[:,1])*100 + 270
	mask_coords[:,2] = mask_coords[:,2]/np.max(mask_coords[:,2])*40 + 2
	mask_coords = mask_coords.astype(np.uint32)
	
	# define the 3D matrix and populate it with the mask coordinates
	mask = np.zeros((np.max(mask_coords,0)+1).tolist())

	for i in range(mask_coords.shape[0]):
		mask[mask_coords[i,0],mask_coords[i,1],mask_coords[i,2]] = 1
		
	# define the 3D matrix for creating the volume from the mesh, extra 20 in x/y will 
	mask_fill = np.zeros([np.max(mask_coords,0)[0]+41,np.max(mask_coords,0)[1]+41,np.max(mask_coords,0)[2]+1])

	for i in range(mask_fill.shape[2]):

		# dilate the pixels
		mask_fill[20:(mask_fill.shape[0]-20),20:(mask_fill.shape[1]-20),i] = mask[:,:,i]
		mask_fill[:,:,i] = cv2.dilate(mask_fill[:,:,i], np.ones((3,3), np.uint8), iterations=3)
		
		# fill the holes
		mask_fill[:,:,i] = binary_fill_holes(mask_fill[:,:,i])
		
		# erode the extra pixels introduced by dilated pixels
		mask_fill[:,:,i] = cv2.erode(mask_fill[:,:,i], np.ones((3,3), np.uint8), iterations=3)  

	# trim back the extra 20 pixel borders added
	mask_fill = mask_fill[20:(mask_fill.shape[0]-20),20:(mask_fill.shape[1]-20),:].astype(np.uint8)

	# convert mask from 0,1 to 0,255, code commented beside it plots it
	mask_fill *= 255 #n=50;plt.imshow(mask_fill[:,:,n]/2+mask[:,:,n]*100,cmap=plt.cm.gray);plt.show()
	
	# reshape to 640x640x44
	mask_new = np.zeros([640,640,44])
	for i in range(mask_fill.shape[2]):
		mask_new[0:mask_fill.shape[0],0:mask_fill.shape[1],i] = mask_fill[:,:,i]
	
	# write image stack to file
	if os.path.isdir(melbourne_patients[n_read]+"/mask"):
		shutil.rmtree(melbourne_patients[n_read]+"/mask")
	
	os.mkdir(melbourne_patients[n_read]+"/mask")
	
	for i in range(mask_new.shape[2]):
		cv2.imwrite(melbourne_patients[n_read]+"/mask/"+"{0:03}".format(i+1)+".tif",mask_new[:,:,i])