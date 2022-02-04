import os
import cv2
import sys
import h5py
import shutil
import plotly
import random
import scipy.io
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import label
import plotly.graph_objects as go
from scipy.spatial import Delaunay
from collections import defaultdict
from scipy.ndimage.filters import uniform_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_fill_holes

def create_folder(file_dir):
	if not os.path.exists(file_dir):
		os.mkdir(file_dir)
		
def simple_alpha_shape_3D(pos, alpha):

	# Compute the alpha shape (concave hull) of a set of 3D points.
	# Parameters:
	#     pos - np.array of shape (n,3) points.
	#     alpha - alpha value.
	# return
	#     outer surface vertex indices, edge indices, and triangle indices


	tetra = Delaunay(pos)

	# Find radius of the circumsphere.
	# By definition, radius of the sphere fitting inside the tetrahedral needs 
	# to be smaller than alpha value
	# http://mathworld.wolfram.com/Circumsphere.html
	tetrapos = np.take(pos,tetra.vertices,axis=0)
	normsq   = np.sum(tetrapos**2,axis=2)[:,:,None]
	ones     = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
	
	a  = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
	Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
	Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
	Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
	c  = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
	r  = np.sqrt(Dx**2+Dy**2+Dz**2-4*a*c)/(2*np.abs(a))

	# Find tetrahedrals
	tetras = tetra.vertices[r<alpha,:]

	# triangles
	TriComb   = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
	Triangles = tetras[:,TriComb].reshape(-1,3)
	Triangles = np.sort(Triangles,axis=1)

	# Remove triangles that occurs twice, because they are within shapes
	TrianglesDict = defaultdict(int)
	for tri in Triangles:
		TrianglesDict[tuple(tri)] += 1
	
	Triangles = np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])

	#edges
	EdgeComb = np.array([(0, 1), (0, 2), (1, 2)])
	Edges    = Triangles[:,EdgeComb].reshape(-1,2)
	Edges    = np.sort(Edges,axis=1)
	Edges    = np.unique(Edges,axis=0)

	# index of points in the original point loud used for the ashape
	Vertices = np.unique(Edges)

	return Vertices,Edges,Triangles
	
def load_nrrd(full_path_filename):
	
	# this function loads .nrrd files into a 3D matrix and outputs it
	# the input is the specified file path
	# the output is the N x A x B for N slices of sized A x B
	
	data = sitk.ReadImage( full_path_filename )							# read in image
	data = sitk.Cast( sitk.RescaleIntensity(data), sitk.sitkUInt8 )		# convert to 8 bit (0-255)
	data = sitk.GetArrayFromImage( data )								# convert to numpy array
	
	return(data)

def smooth3D_interpolate(data,threshold=20,factor=2):
	
	def interpolate_data_z(ImageIn,factor):
		# this function interpolates the 3D data in the z direction depending on the factor
		Nx,Ny,Nz = ImageIn.shape
		x,y,z = np.linspace(1,Nx,Nx),np.linspace(1,Ny,Ny),np.linspace(1,Nz,Nz)
		interp_func = RegularGridInterpolator((x,y,z),ImageIn,method="linear")
		[xi,yi,zi] = np.meshgrid(x,y,np.linspace(1,Nz,factor*Nz),indexing='ij')
		ImageIn = interp_func( np.stack([xi,yi,zi],axis=3) )
		
		return(ImageIn)
	
	# this function interpolates the MRI and smoothes it in 3D
	data[data>=1] = 1;
	data[data!=1] = 0
	data[data==1] = 50
	data = interpolate_data_z(data,factor)
	data = uniform_filter(data,5)
	data[data <  threshold] = 0
	data[data >= threshold] = 50
	data = data//50
	
	return(data)
	
def compute_utah_PV(endocardium,wall,endocardium_no_veins):
	
	# subtract wall and endo(without vein) to get PV pixels
	veins = endocardium + wall - endocardium_no_veins
	veins[veins>1]   = 1
	veins[wall!=0] = 0
	veins[endocardium_no_veins!=0] = 0

	# use cross shape structure to find connected components in the 3D mask
	mask_reg,mask_lab = label(veins,structure=np.array([[[0,0,0],[0,1,0],[0,0,0]],
														[[0,1,0],[1,1,1],[0,1,0]],
														[[0,0,0],[0,1,0],[0,0,0]]]))
	
	# compute size of components and find which ones are the biggest (greater than mean size)
	reg_size = np.array([np.sum(mask_reg[mask_reg == i]) for i in np.unique(mask_reg)])
	mean_reg = np.mean(reg_size)
	
	# sort the connected regions by size
	mask_ind_sorted = np.argsort(reg_size)[::-1] #[(len(reg_size)-4):len(reg_size)]
	mask_ind_sorted = mask_ind_sorted[mask_ind_sorted>0]

	# estimate mitral valve location, then get largest regions with are not masked out by the mitral valve region
	_,_,z   = np.where(endocardium==1)
	x,y     = np.where(endocardium[:,:,np.min(z)]==1)
	ind_max = np.argmin(np.sqrt(np.abs(0-x)**2+np.abs(640-y)**2))
	MV_ind  = np.array([x[ind_max]+15,y[ind_max]-15,np.min(z)+10])

	PV_ind,i = [],0
	while len(PV_ind) < 4 and i < len(mask_ind_sorted):
		mask_temp = (mask_reg == mask_ind_sorted[i]).astype(np.int64)
		com_temp  = center_of_mass(mask_temp)
		
		if not((com_temp[0] < MV_ind[0] and com_temp[2] < MV_ind[2]) or (com_temp[1] > MV_ind[1] and com_temp[2] < MV_ind[2])):
			PV_ind.append(mask_ind_sorted[i])
			
		i += 1
	
	# if there were no PV detected, manually define these regions
	if len(PV_ind) == 0:
		print("manual PV")
		x,y,z   = np.where(endocardium==1)
		utah_l_names = np.array(["RSPV","RIPV","LIPV"])
		utah_landmark = np.zeros([3,3])
		
		ind_temp = np.argmin(np.abs(0-x)+np.abs(0-y)+np.abs(22-z))
		utah_landmark[:,0] = [x[ind_temp],y[ind_temp],z[ind_temp]]
		
		ind_temp = np.argmin(np.abs(640-x)+np.abs(0-y)+np.abs(22-z))
		utah_landmark[:,1] = [x[ind_temp],y[ind_temp],z[ind_temp]]
		
		ind_temp = np.argmin(np.abs(640-x)+np.abs(640-y)+np.abs(22-z))
		utah_landmark[:,2] = [x[ind_temp],y[ind_temp],z[ind_temp]]
		
		return(utah_landmark,utah_l_names,mask_reg)
		
	# loop through the biggest connected components and define them in the mask as negative values
	for i in range(len(PV_ind)):
		mask_reg[mask_reg == PV_ind[i]] = -1*(i+1)

	# remove all positive values and make the newly define negative valued regions positive
	mask_reg[mask_reg>0] = 0
	mask_reg *= -1

	# define center of mass
	com = center_of_mass(endocardium)
	utah_landmark = np.zeros([3,4])/0

	# define points in each individual connected component (largest found), the furtherest point in the component is a potential vein
	for i in range(0,len(np.unique(mask_reg))-1):	
		if np.sum(mask_reg==(i+1)) > mean_reg*0.1:
			x,y,z   = np.where(mask_reg==(i+1))
			ind_max = np.argmax(np.abs(com[0]-x)+np.abs(com[1]-y)+np.abs(com[2]-z))
			utah_landmark[:,i] = [x[ind_max],y[ind_max],z[ind_max]]

	# define 4 potential place holders for the 4 PVs
	utah_l_names = np.array(["xxxx","xxxx","xxxx","xxxx"])
	
	# threshold the mask and fine the center of mass of the current potential PVs identified
	temp         = np.copy(mask_reg)
	temp[temp>0] = 1
	com_vein     = center_of_mass(temp)

	# the vein closet to (0,0) and closer to (0,0) than the PV center of mass is RSPV
	ind = np.nanargmin(((utah_landmark[0,:] - 0)**2 + (utah_landmark[1,:] - 0)**2)**0.5)
	if utah_landmark[:,ind][0] < com_vein[0] and utah_landmark[:,ind][1] < com_vein[1]:
		utah_l_names[ind] = "RSPV"
	
	# the vein closet to (640,0) and closer to (640,0) than the PV center of mass is RIPV
	ind  = np.nanargmin(((utah_landmark[0,:] - endocardium.shape[0])**2 + (utah_landmark[1,:] - 0)**2)**0.5)
	if utah_landmark[:,ind][0] > com_vein[0] and utah_landmark[:,ind][1] < com_vein[1]:
		utah_l_names[ind] = "RIPV"
	
	# the vein closet to (0,640) and closer to (0,640) than the PV center of mass is LSPV
	ind = np.nanargmin(((utah_landmark[0,:] - 0)**2 + (utah_landmark[1,:] - endocardium.shape[0])**2)**0.5)
	if utah_landmark[:,ind][0] < com_vein[0] and utah_landmark[:,ind][1] > com_vein[1]:
		utah_l_names[ind] = "LSPV"

	# the vein closet to (640,640) and closer to (640,640) than the PV center of mass is LSPV		
	ind = np.nanargmin(((utah_landmark[0,:] - endocardium.shape[0])**2 + (utah_landmark[1,:] - endocardium.shape[1])**2)**0.5)
	if utah_landmark[:,ind][0] > com_vein[0] and utah_landmark[:,ind][1] > com_vein[1]:
		utah_l_names[ind] = "LIPV"

	# remove the place holders which are unallocated
	delete_cols = np.where(utah_l_names == 'xxxx')[0]

	# remove these columns from the PV coord array and name array
	utah_landmark    = np.delete(utah_landmark,delete_cols,axis=1)
	utah_l_names     = np.delete(utah_l_names,delete_cols)
	
	return(utah_landmark,utah_l_names,mask_reg)
	
def cartesian2polar(cart1,cart2,ref1,ref2):
	# given set of points x,y and their reference point x',y'
	# 		this function returns their polar coordinates w.r.t the reference point
	# 		note the the inputs of this is the inverse of the outputs to the next function
	# 		to get x,y output in the next function, the input to this function should be y,x
	return(np.hypot(cart1-ref1,cart2-ref2), np.arctan2(cart1-ref1,cart2-ref2))

def polar2cartesian(polar_d,polar2_r,ref1,ref2):
	# given a polar coordinate, compute its cartesian version w.r.t the reference point
	# 		note the the outputs of this is the inverse of the inputs to the previous function
	# 		to get x,y output, the input to the function above should be y,x
	return(ref1+polar_d*np.cos(polar2_r), ref2+polar_d*np.sin(polar2_r))

def PointCloud2Raster(input_coords,out_shape):

	# define the 3D matrix and populate it with the mask coordinates
	mask = np.zeros(list(out_shape))

	for i in range(input_coords.shape[0]):
		mask[int(input_coords[i,0]),int(input_coords[i,1]),int(input_coords[i,2])] = 1
		
	# define the 3D matrix for creating the volume from the mesh, extra 20 in x/y padding 
	mask_fill = np.zeros([mask.shape[0]+60,mask.shape[1]+60,mask.shape[2]])

	for i in range(mask_fill.shape[2]):

		# dilate the pixels
		mask_fill[30:(mask_fill.shape[0]-30),30:(mask_fill.shape[1]-30),i] = mask[:,:,i]
		mask_fill[:,:,i] = cv2.dilate(mask_fill[:,:,i], np.ones((3,3), np.uint8), iterations=5)
		
		# fill the holes
		mask_fill[:,:,i] = binary_fill_holes(mask_fill[:,:,i])
		
		# erode the extra pixels introduced by dilated pixels
		mask_fill[:,:,i] = cv2.erode(mask_fill[:,:,i], np.ones((3,3), np.uint8), iterations=5)  

	# trim back the extra 20 pixel borders added
	mask_fill = mask_fill[30:(mask_fill.shape[0]-30),30:(mask_fill.shape[1]-30),:].astype(np.uint8)
	
	return(mask_fill)

def raster2mesh(mask_temp):

		# make a copy of the input
		mask = np.copy(mask_temp)

		# interpolate and create a thin outer shell
		mask_temp = np.zeros_like(mask)
		for i in range(mask.shape[2]):
			mask_temp[:,:,i] = cv2.erode(np.uint8(mask[:,:,i]),np.ones((3,3),np.uint8),iterations=1)

		# compute mesh
		x,y,z = np.where((mask - mask_temp) == 1)
		coords = np.array([x,y,z]).T
		vert,edge,triang = simple_alpha_shape_3D(coords,4)
		
		return(coords,triang)
		
def plot_raster_as_mesh(mask,path,PV_loc,PV_names,out_name,compute_dist=True,save=True):

	# compute delaunay triangulation to visualize surface as mesh
	a_coords,a_triang = raster2mesh(mask)
	
	# plot data
	fig = go.Figure(data=[go.Mesh3d(x=a_coords[:,1],y=a_coords[:,0],z=a_coords[:,2],
							        i=a_triang[:,1],j=a_triang[:,0],k=a_triang[:,2],color='grey',opacity=0.1)])
									
	if compute_dist:
		# compute distance of each point to the surface mesh, use as colorscale for plotting
		path_dist = []
		for i in range(path.shape[0]):
			path_dist.append(np.min(((a_coords[:,0]-path[i,0])**2 + (a_coords[:,1]-path[i,1])**2 + (a_coords[:,2]-path[i,2])**2)**0.5))
									
		fig.add_trace(go.Scatter3d(x=path[:,1],y=path[:,0],z=path[:,2],
						           mode='markers',marker=dict(size=2,color=path_dist,colorscale='Jet',opacity=0.8)))
	else:
		fig.add_trace(go.Scatter3d(x=path[:,1],y=path[:,0],z=path[:,2],
						           mode='markers',marker=dict(size=2,color="#8a2be2")))
	
	fig.add_trace(go.Scatter3d(x=PV_loc[1,:],y=PV_loc[0,:],z=PV_loc[2,:],
						       mode='markers+text',marker=dict(size=7.5,color="lightpink",symbol='diamond'),
						       text=PV_names,textfont={"size":30,"color":"red"},textposition="top center"))
	fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
					  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=0.75),
								 xaxis=dict(showbackground=False),xaxis_title='', #,tickvals=[],ticktext=[]
								 yaxis=dict(showbackground=False),yaxis_title='', #,tickvals=[],ticktext=[]
								 zaxis=dict(showbackground=True),zaxis_title='',  #,tickvals=[],ticktext=[]
					            )
					  )

	if save:
		plotly.offline.plot(fig,filename=out_name+'.html',auto_open=False)
	else:
		fig.show()

def utah_plot_raster_as_mesh(mask,mask_PV_temp,utah_PV,utah_PV_names,save=True,save_name_path=""):

	# compute delaunay triangulation to visualize surface as mesh
	a_coords,a_triang = raster2mesh(mask)

	# compute delaunay triangulation to visualize surface as mesh for the PV veins exclusively
	mask_PV_temp[mask_PV_temp>0] = 1
	pv_coords,pv_triang = raster2mesh(mask_PV_temp)
	
	# plot
	fig = go.Figure(data=[go.Mesh3d(x=a_coords[:,0],y=a_coords[:,1],z=a_coords[:,2],color='grey',
									i=a_triang[:,0],j=a_triang[:,1],k=a_triang[:,2],
									opacity=0.1,name="LA Surface")])
	fig.add_trace(go.Mesh3d(x=pv_coords[:,0],y=pv_coords[:,1],z=pv_coords[:,2],color='mediumpurple',
							i=pv_triang[:,0],j=pv_triang[:,1],k=pv_triang[:,2],opacity=0.075,name="Pulmonary Veins"))
	fig.add_trace(go.Scatter3d(x=a_coords[:,0],y=a_coords[:,1],z=a_coords[:,2],
							   mode='markers',marker=dict(size=1,color="black",opacity=0.1)))
	fig.add_trace(go.Scatter3d(x=utah_PV[0,:],y=utah_PV[1,:],z=utah_PV[2,:]*(np.max(a_coords[:,2])//45+1),
							   mode='markers+text',marker=dict(size=5,color="lightpink",symbol='diamond'),
							   text=utah_PV_names,textfont={"size":20,"color":"red"},textposition="top center"))
	fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
					  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=2),
								 xaxis=dict(showbackground=False),xaxis_title='', #,tickvals=[],ticktext=[]
								 yaxis=dict(showbackground=False),yaxis_title='', #,tickvals=[],ticktext=[]
								 zaxis=dict(showbackground=True),zaxis_title='',  #,tickvals=[],ticktext=[]
								)
					  )
						
	if save:
		plotly.offline.plot(fig,filename=save_name_path+'.html',auto_open=False)
	else:
		fig.show()