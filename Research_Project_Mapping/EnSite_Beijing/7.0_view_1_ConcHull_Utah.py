from utils import *
import scipy.io
import os
import numpy as np
import plotly
import plotly.graph_objects as go
import cv2
import matplotlib.pyplot as plt

n = 0
utah_patient = "1421pre" # "1160pre" "1421pre" "0100pre" "02914mo" "10833mo" "13878mo" "08866mo"

beijing_patients = ["BaoZhongzhi",   # 0
                    "GaoXing",       # 1
                    "JiangTongjie",  # 2
                    "LiWenhai",      # 3
                    "ShenPinsheng",  # 4
                    "XinhuaWang",    # 5
                    "ZhangNengqin",  # 6
                    "ZhangWuhui",    # 7
                    "ZhenLixing",    # 8
                    "ZhiJianyou"]    # 9

# load data
laendo          = np.rollaxis(load_nrrd("Utah_Sample/"+utah_patient+"/laendo.nrrd")//255,0,3)
lawall          = np.rollaxis(load_nrrd("Utah_Sample/"+utah_patient+"/lawall.nrrd")//255,0,3)
laendo_no_veins = np.rollaxis(load_nrrd("Utah_Sample/"+utah_patient+"/laendo_no_veins.nrrd")//255,0,3)

# get PVs from Utah
utah_PV,utah_PV_names,mask_PV = compute_utah_PV(laendo,lawall,laendo_no_veins)

utah_com = center_of_mass(laendo)

if len(utah_PV_names) <= 2:
	utah_PV_com = np.array([utah_com[0],utah_com[1],np.mean(utah_PV,1)[2]])
else:
	utah_PV_com = np.mean(utah_PV,1)

# load bejing data obtained via ensite software during ablation
dat = scipy.io.loadmat("C:/Users/Administrator/Desktop/CatheterPath/2010_BeiJing_PointCloud/"+beijing_patients[n]+"/"+beijing_patients[n]+".mat")

coords    = dat['coord']
path      = dat['path']
path_dist = dat['pathdist']
landmark  = np.stack(dat['landmark'][0][0],axis=1)[:,:,0]
l_names   = np.array(dat['landmark'][0].dtype.names)

### --------------------------------------------------------------------------------------------------- COMPUTE TRANSFORMATIONS 1-6
# # # 1 - flip beijing data by swapping x and y axis to get Utah's orientation
landmark_flip = np.copy(landmark)
landmark_flip[0,:],landmark_flip[1,:] = landmark_flip[1,:],landmark_flip[0,:].copy()

# find normal vectors for the PVs for both beijing and utah, in the correct direction (away from LA COM)
ensite_PV_com,ensite_com = np.mean(landmark_flip,1),np.mean(coords,0)
PV_norm = np.cross(landmark_flip[:,l_names=="LIPV"][:,0] - landmark_flip[:,l_names=="RSPV"][:,0],
				   landmark_flip[:,l_names=="LIPV"][:,0] - landmark_flip[:,l_names=="RIPV"][:,0])
PV_norm = PV_norm/np.linalg.norm(PV_norm)

# if vector pointing towards LA COM, reverse direction of vector
if np.linalg.norm(ensite_PV_com - ensite_com) > np.linalg.norm(ensite_PV_com + PV_norm*10 - ensite_com):
	PV_norm *= -1 # if vector pointing towards LA COM, reverse direction of vector

PV_norm_utah = np.array([0,0,1])

# # # 2 - rotate bejing data along y/z axis (x fixed) so that veins are in orientation that matches the utah PV, align PVs vertically up
landmark_rot = np.copy(landmark_flip)
rot_vec      = landmark_rot - ensite_PV_com[:,None]

polar_ensite_x = cartesian2polar(PV_norm[2],PV_norm[1],0,0)
polar_utah_x   = cartesian2polar(PV_norm_utah[2],PV_norm_utah[1],0,0)
rotate_x       = polar_utah_x[1] - polar_ensite_x[1]

PV_norm[1],PV_norm[2] = polar2cartesian(polar_ensite_x[0],polar_ensite_x[1]+rotate_x,0,0)

polar_x = cartesian2polar(rot_vec[2,:],rot_vec[1,:],0,0)
rot_vec[1],rot_vec[2] = polar2cartesian(polar_x[0],polar_x[1]+rotate_x,0,0)

# # # 3 - rotate bejing data along x/z (y fixed) axis so that veins are in orientation that matches the utah PV, align PVs vertically up
polar_ensite_y = cartesian2polar(PV_norm[2],PV_norm[0],0,0)
polar_utah_y   = cartesian2polar(PV_norm_utah[2],PV_norm_utah[0],0,0)
rotate_y       = polar_utah_y[1] - polar_ensite_y[1]

PV_norm[0],PV_norm[2] = polar2cartesian(polar_ensite_y[0],polar_ensite_y[1]+rotate_y,0,0)

polar_y = cartesian2polar(rot_vec[2,:],rot_vec[0,:],0,0)
rot_vec[0],rot_vec[2] = polar2cartesian(polar_y[0],polar_y[1]+rotate_y,0,0)

# # # 4 - rotate data along x/y (z fixed) so the orientation of the PVs are the same as Utah's, align PVs vertically up
landmark_rot = ensite_PV_com[:,None] + rot_vec # perform rotation in x and y

ensite_PV_com = np.mean(landmark_rot,1)

polar_ensite_z = cartesian2polar(landmark_rot[1,:],landmark_rot[0,:],ensite_PV_com[1],ensite_PV_com[0])
polar_utah_z   = cartesian2polar(utah_PV[1,0],utah_PV[0,0],utah_PV_com[1],utah_PV_com[0])
rotate_z       = polar_utah_z[1] - polar_ensite_z[1][np.where(l_names == utah_PV_names[0])[0]]

landmark_rot[0],landmark_rot[1] = polar2cartesian(polar_ensite_z[0],polar_ensite_z[1]+rotate_z,ensite_PV_com[0],ensite_PV_com[1])

# # # 5 - 2nd iteration rotate data along x/z (y fixed), align PVs with Utah norm
if utah_PV.shape[1] >= 2:
	if utah_PV.shape[1] >= 3:
		PV1  = utah_PV[:,0]
		ind2 = np.argmax(np.mean(np.abs(PV1[:,None] - utah_PV),0))
		PV2  = utah_PV[:,ind2]
		PV3  = utah_PV[:,[i for i in range(1,utah_PV.shape[1]) if i != ind2][0]]
	elif utah_PV.shape[1] == 2:
		PV1 = utah_PV_com
		PV2 = utah_PV[:,0]
		PV3 = utah_PV[:,1]

	PV_norm_utah = np.cross(PV1-PV2,PV1-PV3)
	PV_norm_utah = PV_norm_utah/np.linalg.norm(PV_norm_utah)
	PV_norm_utah *= np.sign(PV_norm_utah[2]) # if z is negative, flip to positive
else:
	PV_norm_utah = np.array([0,0,1])

rot_vec  = landmark_rot - ensite_PV_com[:,None]

polar_ensite_y2 = cartesian2polar(PV_norm[2],PV_norm[0],0,0)
polar_utah_y2   = cartesian2polar(PV_norm_utah[2],PV_norm_utah[0],0,0)
rotate_y2       = polar_utah_y2[1] - polar_ensite_y2[1]

PV_norm[0],PV_norm[2] = polar2cartesian(polar_ensite_y2[0],polar_ensite_y2[1]+rotate_y2,0,0)

polar_y2              = cartesian2polar(rot_vec[2,:],rot_vec[0,:],0,0)
rot_vec[0],rot_vec[2] = polar2cartesian(polar_y2[0],polar_y2[1]+rotate_y2,0,0)

landmark_rot = ensite_PV_com[:,None] + rot_vec 

# # # 6 - linear translation to match one of the PVs
translate = utah_PV_com - np.mean(landmark_rot,1)
landmark_translate = np.copy(landmark_rot) + np.array(translate)[:,None]

### --------------------------------------------------------------------------------------------------- APPLY TRANSFORMATIONS 1-6
# # # 1 - swap x/y axis
path[:,0],path[:,1]     = path[:,1],path[:,0].copy()
coords[:,0],coords[:,1] = coords[:,1],coords[:,0].copy()

# # # 2 - rotate while fixing x axis
polar_x             = cartesian2polar(path[:,2],path[:,1],ensite_PV_com[2],ensite_PV_com[1])
path[:,1],path[:,2] = polar2cartesian(polar_x[0],polar_x[1]+rotate_x,ensite_PV_com[1],ensite_PV_com[2])

polar_x             = cartesian2polar(coords[:,2],coords[:,1],ensite_PV_com[2],ensite_PV_com[1])
coords[:,1],coords[:,2] = polar2cartesian(polar_x[0],polar_x[1]+rotate_x,ensite_PV_com[1],ensite_PV_com[2])

# # # 3 - rotate while fixing y axis
polar_y             = cartesian2polar(path[:,2],path[:,0],ensite_PV_com[2],ensite_PV_com[0])
path[:,0],path[:,2] = polar2cartesian(polar_y[0],polar_y[1]+rotate_y,ensite_PV_com[0],ensite_PV_com[2])

polar_y             = cartesian2polar(coords[:,2],coords[:,0],ensite_PV_com[2],ensite_PV_com[0])
coords[:,0],coords[:,2] = polar2cartesian(polar_y[0],polar_y[1]+rotate_y,ensite_PV_com[0],ensite_PV_com[2])

# # # 4 - rotate while fixing z axis
polar_z             = cartesian2polar(path[:,1],path[:,0],ensite_PV_com[1],ensite_PV_com[0])
path[:,0],path[:,1] = polar2cartesian(polar_z[0],polar_z[1]+rotate_z,ensite_PV_com[0],ensite_PV_com[1])

polar_z             = cartesian2polar(coords[:,1],coords[:,0],ensite_PV_com[1],ensite_PV_com[0])
coords[:,0],coords[:,1] = polar2cartesian(polar_z[0],polar_z[1]+rotate_z,ensite_PV_com[0],ensite_PV_com[1])

# # # 5 - 2nd iteration rotate while fixing y axis
polar_y = cartesian2polar(path[:,2],path[:,0],ensite_PV_com[2],ensite_PV_com[0])
path[:,0],path[:,2] = polar2cartesian(polar_y[0],polar_y[1]+rotate_y2,ensite_PV_com[0],ensite_PV_com[2])

polar_y = cartesian2polar(coords[:,2],coords[:,0],ensite_PV_com[2],ensite_PV_com[0])
coords[:,0],coords[:,2] = polar2cartesian(polar_y[0],polar_y[1]+rotate_y2,ensite_PV_com[0],ensite_PV_com[2])

# # # 6 - linear translation
path    += translate
coords  += translate

### --------------------------------------------------------------------------------------------------- COMPUTE TRANSFORMATIONS 7
# # # 7 - linear scaling to scale the beijing geometry with the utah one
landmark_scale,ensite_PV_com = np.copy(landmark_translate),np.mean(landmark_translate,1)

ensite_size = np.mean(np.abs(landmark_scale[:,np.array(["PV" in s for s in l_names])] - ensite_PV_com[:,None]),1)
utah_size   = np.mean(np.abs(utah_PV - utah_PV_com[:,None]),1)

scale_xy = utah_size[:2]/ensite_size[:2]
scale_z  = (np.max(np.where(laendo == 1)[2]*2) - np.min(np.where(laendo == 1)[2]*2))/(np.max(coords,0)[2] - np.min(coords,0)[2])
scale    = np.array([scale_xy[0],scale_xy[1],scale_z])

landmark_scale = (landmark_scale - utah_PV_com[:,None])*scale[:,None] + utah_PV_com[:,None]

### --------------------------------------------------------------------------------------------------- APPLY TRANSFORMATIONS 7
# # # 7 - linear scaling
path     = (path - utah_PV_com)*scale + utah_PV_com
coords   = (coords - utah_PV_com)*scale + utah_PV_com
landmark = np.copy(landmark_scale)

### --------------------------------------------------------------------------------------------------- PROJECT PATH ONTO LGE-MRI
# expand coords to make it go outside utah path
cath_com  = np.mean(coords,0)
path = (path - cath_com)*100
path = path + cath_com

# move beijing center of mass to utah center of mass
cath_com  = np.mean(path,0)
xy_diff   = utah_com - cath_com
cath_path = path + [xy_diff[0],xy_diff[1],0]

# project catheter path to utah data surface
utah_path = []

for i in range(cath_path.shape[0]):
	
	# find closest point in catheter path to beijing surface
	path_point   = cath_path[i,:]
	ind          = np.argmin(np.sum((path_point-coords)**2,1)**0.5)
	ensite_point = coords[ind,:]

	# compte vector passing through path point and COM
	vec_line = path_point - utah_com
	vec_line = vec_line/np.max(np.abs(vec_line))*320

	# find all utah endo points along the vector line
	max_n = np.ceil(np.max(np.abs(vec_line)))
	max_p = utah_com + vec_line
	vec_x = np.linspace(int(utah_com[0]),int(max_p[0]),int(max_n))
	vec_y = np.linspace(int(utah_com[1]),int(max_p[1]),int(max_n))
	vec_z = np.linspace(int(utah_com[2]),int(max_p[2]),int(max_n))
	
	vec_line = np.stack([vec_x,vec_y,vec_z],axis=1)
	x,y,z    = laendo.shape
	vec_line = vec_line[np.logical_and(vec_line[:,0] < x,vec_line[:,0] > 0),:]
	vec_line = vec_line[np.logical_and(vec_line[:,1] < y,vec_line[:,1] > 0),:]
	vec_line = vec_line[np.logical_and(vec_line[:,2] < z,vec_line[:,2] > 0),:]

	# find furthest point along vec_line from utah_com that is still contaiend within utah laendo
	utah_in_vec = [laendo[int(vec_line[k,0]),int(vec_line[k,1]),int(vec_line[k,2])] for k in range(vec_line.shape[0])]
	utah_in_vec = np.max(np.where(np.array(utah_in_vec) == 1)[0])
	
	utah_point = vec_line[utah_in_vec,:]

	# compute vector from surface to catheter path point
	ensite_vec = path_point - ensite_point
	ensite_vec = ensite_vec/np.linalg.norm(ensite_vec)
	
	# apply the vector from utah surface to obtain utah catheter path, scale the distance according to volume
	if np.linalg.norm(utah_point + ensite_vec - utah_com) > np.linalg.norm(utah_point - ensite_vec - utah_com):
		utah_path.append(utah_point - ensite_vec)
	else:
		utah_path.append(utah_point + ensite_vec)

a_coords = np.array(utah_path)

#pd.DataFrame(a_coords).to_csv("1421pre_projected_path.csv",header=["X","Y","Z"],index=False)

### Generate More Things to Plot
_,_,a_triang = simple_alpha_shape_3D(a_coords,200)

if True:
	fig = go.Figure(data=[go.Mesh3d(x=a_coords[:,1],y=a_coords[:,0],z=a_coords[:,2],color='grey',
							        i=a_triang[:,1],j=a_triang[:,0],k=a_triang[:,2],
							        opacity=0.5,name="LA Surface")])
	fig.add_trace(go.Scatter3d(x=a_coords[:,1],y=a_coords[:,0],z=a_coords[:,2],
						       mode='markers',marker=dict(size=2,color="black",opacity=0.75)))
	fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
					  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=0.75),
					             xaxis=dict(showbackground=False),xaxis_title='', #,tickvals=[],ticktext=[]
					             yaxis=dict(showbackground=False),yaxis_title='', #,tickvals=[],ticktext=[]
					             zaxis=dict(showbackground=False),zaxis_title='',  #,tickvals=[],ticktext=[]
					            )
					  )
	fig.show()
