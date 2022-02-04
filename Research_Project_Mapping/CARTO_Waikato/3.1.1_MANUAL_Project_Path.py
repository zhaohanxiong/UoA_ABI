from utils import *

### Set Origin Path
n = 5

### Load files
# Waikato Mesh --------------------------------------------------------------------------------------------------
os.chdir("C:/Users/zxio506/Desktop/Atria_Data/CatheterPath/2020_Waikato_PointCloud")

# patient folders
waikato_patients = ["Patient_0002-Richmond_Michelle", # 0
					"Patient_0003-DEEGAN-Janet",      # 1
					"Patient_0004-CRAIG_Leslie",      # 2
					"Patient_0005-SIMON_Barry",       # 3
					"Patient_0006-CRONIN_Paul",       # 4
					"Patient_00011-WATKINS_Catheryn"] # 5

# load data
dat = scipy.io.loadmat(waikato_patients[n]+"/catheter.mat")

coords        = dat['coord']
p_conchull    = dat['ashape_x']
tri_conchull  = dat['ashape_tri'].astype(np.int32) - 1 # python is 0 indexed so need to convert this from R
path          = dat['path']
path_dist     = dat['pathdist']

# Waikato LGE-MRI ------------------------------------------------------------------------------------------------
os.chdir("C:/Users/zxio506/Desktop/Atria_Data/Waikato")

# patient folders
waikato_patient_LGEMRI = ["","00003","00004","00005","","000011"]

# load data
lab = np.zeros([640,640,44])
f2  = os.listdir(waikato_patient_LGEMRI[n]+"/label")

for i in range(lab.shape[2]):
	lab[:,:,i] = cv2.imread(waikato_patient_LGEMRI[n]+"/label/"+f2[i],cv2.IMREAD_GRAYSCALE)
	
lab = (lab==4).astype(np.int8)

# sample 3D points from the mesh to produce point cloud
def raster2mesh(mask_temp):

	mask = np.copy(mask_temp)

	# interpolate and create a thin outer shell
	mask_temp = np.zeros_like(mask)
	for i in range(mask.shape[2]):
		mask_temp[:,:,i] = cv2.erode(np.uint8(mask[:,:,i]),np.ones((5,5),np.uint8),iterations=1)

	# compute mesh
	x,y,z = np.where((mask - mask_temp) == 1)
	coords = np.array([x,y,z]).T
	vert,edge,triang = simple_alpha_shape_3D(coords,2)
	
	return(coords,triang)

a_coords,a_triang = raster2mesh(lab)

lgemri_com = np.mean(a_coords,0)

### Match Geometry
# swap some axis
new_p_conchull      = np.copy(p_conchull)
new_p_conchull[:,0] = p_conchull[:,1]
new_p_conchull[:,1] = p_conchull[:,0]
p_conchull          = np.copy(new_p_conchull)

# rotate
rotate_degree = 0.5*np.pi
com           = center_of_mass(lab)

polar_p_conchull                = cartesian2polar(p_conchull[:,2],p_conchull[:,0],com[2],com[0])
p_conchull[:,0],p_conchull[:,2] = polar2cartesian(polar_p_conchull[0],polar_p_conchull[1]+rotate_degree,com[0],com[2])

# find LGE-MRI coordinates
min_x,max_x = np.min(a_coords[:,0]),np.max(a_coords[:,0])
min_y,max_y = np.min(a_coords[:,1]),np.max(a_coords[:,1])
min_z,max_z = np.min(a_coords[:,2]),np.max(a_coords[:,2])

# rescale mesh to start from (0,0,0)
p_conchull[:,0] = p_conchull[:,0] - np.min(p_conchull[:,0])
p_conchull[:,1] = p_conchull[:,1] - np.min(p_conchull[:,1])
p_conchull[:,2] = p_conchull[:,2] - np.min(p_conchull[:,2])

# resize to same dims as utah
p_conchull[:,0] = p_conchull[:,0]/np.max(p_conchull[:,0])*(max_x-min_x)
p_conchull[:,1] = p_conchull[:,1]/np.max(p_conchull[:,1])*(max_y-min_y)
p_conchull[:,2] = p_conchull[:,2]/np.max(p_conchull[:,2])*(max_z-min_z)

# compute COM
conchull_com    = np.mean(p_conchull,0)

# path: swap some axis
new_path      = np.copy(path)
new_path[:,0] = path[:,1]
new_path[:,1] = path[:,0]
path          = np.copy(new_path)

# path: rotate
polar_path          = cartesian2polar(path[:,2],path[:,0],com[2],com[0])
path[:,0],path[:,2] = polar2cartesian(polar_path[0],polar_path[1]+rotate_degree,com[0],com[2])

polar_path          = cartesian2polar(path[:,1],path[:,0],com[1],com[0])
path[:,0],path[:,1] = polar2cartesian(polar_path[0],polar_path[1]+0.3,com[0],com[1])

# path: rescale to start from (0,0,0)
path[:,0] = path[:,0] - np.min(path[:,0])
path[:,1] = path[:,1] - np.min(path[:,1])
path[:,2] = path[:,2] - np.min(path[:,2])

# path: resize to same dims as utah, and move to same location
path[:,0] = path[:,0]/np.max(path[:,0])*(max_x-min_x) + (com[0] - conchull_com[0])
path[:,1] = path[:,1]/np.max(path[:,1])*(max_y-min_y) + (com[1] - conchull_com[1])
path[:,2] = path[:,2]/np.max(path[:,2])*(max_z-min_z) + (com[2] - conchull_com[2])

# modify section of points if necessary
#new_path = path[np.logical_and(np.logical_and(path[:,0]>350,path[:,1]>375),path[:,2]>25),:]
new_path = path[path[:,0]<340,:]
new_path[:,2] += 6

#new_path2 = np.copy(new_path)
#polar_path          = cartesian2polar(new_path2[:,1],new_path2[:,0],370,340)
#new_path2[:,0],new_path2[:,1] = polar2cartesian(polar_path[0],polar_path[1]-np.pi*0.5,370,340)

path = np.concatenate((path,new_path))

### Manually Adjust Position
#path = (path - lgemri_com)*0.9
#path = path + lgemri_com

# min_x max_x   min_y max_y   min_z max_z lgemri_com

# ------------------------------------------- Adjust X Position
#path[:,0] -= lgemri_com[0]
#path[:,0] *= 1.2
#path[:,0] += lgemri_com[0]

#path[:,0] -= max_x
#path[:,0] *= 1
#path[:,0] += max_x

#path[:,0] -= min_x
#path[:,0] *= 1.1
#path[:,0] += min_x

#path[:,0] += 12

# ------------------------------------------- Adjust Y Position
#path[:,1] -= lgemri_com[1]
#path[:,1] *= 1.2
#path[:,1] += lgemri_com[1]

#path[:,1] -= min_z
#path[:,1] *= 1.1
#path[:,1] += min_z

#path[:,1] -= max_y
#path[:,1] *= 1.1
#path[:,1] += max_y

#path[:,1] -= 10

# ------------------------------------------- Adjust Z Position
#path[:,2] -= lgemri_com[2]
#path[:,2] *= 1.2
#path[:,2] += lgemri_com[2]

path[:,2] -= min_z
path[:,2] *= 1.1
path[:,2] += min_z

#path[:,2] -= max_z
#path[:,2] *= 1.1
#path[:,2] += max_z

#path[:,2] += 1


waikato_path = np.array(path)

# ### Project Path
if True: # True False

	# waikato center of mass
	waikato_com = np.array(center_of_mass(lab))

	# project catheter path to waikato data surface
	waikato_path = []
	for i in range(path.shape[0]):
		
		# find closest point in catheter path to carto surface
		path_point  = path[i,:]
		ind         = np.argmin(np.sum((path_point-coords)**2,1)**0.5)
		carto_point = coords[ind,:]

		if int(path_point[2]) < 44 and lab[int(path_point[0]),int(path_point[1]),int(path_point[2])] == 1:
			waikato_path.append(path_point)
		else:
		
			# compte vector passing through path point and COM
			vec_line = path_point - waikato_com
			vec_line = vec_line/np.max(np.abs(vec_line))*320

			# find all waikato endo points along the vector line
			max_n = np.ceil(np.max(np.abs(vec_line)))
			max_p = waikato_com + vec_line
			vec_x = np.linspace(waikato_com[0],max_p[0],max_n)
			vec_y = np.linspace(waikato_com[1],max_p[1],max_n)
			vec_z = np.linspace(waikato_com[2],max_p[2],max_n)

			vec_line = np.stack([vec_x,vec_y,vec_z],axis=1)
			x,y,z    = lab.shape
			vec_line = vec_line[np.logical_and(vec_line[:,0] < x,vec_line[:,0] > 0),:]
			vec_line = vec_line[np.logical_and(vec_line[:,1] < y,vec_line[:,1] > 0),:]
			vec_line = vec_line[np.logical_and(vec_line[:,2] < z,vec_line[:,2] > 0),:]

			# find furthest point along vec_line from waikato_com that is still contaiend within waikato laendo
			waikato_in_vec = [lab[int(vec_line[k,0]),int(vec_line[k,1]),int(vec_line[k,2])] for k in range(vec_line.shape[0])]
			waikato_in_vec = np.max(np.where(np.array(waikato_in_vec) == 1)[0])

			waikato_point = vec_line[waikato_in_vec,:]

			# compute vector from surface to catheter path point
			carto_vec = path_point - carto_point
			carto_vec = carto_vec/np.linalg.norm(carto_vec)

			# apply the vector from waikato surface to obtain waikato catheter path
			if np.linalg.norm(waikato_point + carto_vec - utah_com) > np.linalg.norm(waikato_point - carto_vec - utah_com):
				waikato_path.append(waikato_point - carto_vec)
			else:
				waikato_path.append(waikato_point + carto_vec)


	waikato_path = np.array(waikato_path)

	

### Plot
os.chdir("C:/Users/zxio506/Desktop/Atria_Data/CatheterPath/2020_Waikato_PointCloud")

if True:
	fig = go.Figure(data=[go.Mesh3d(x=a_coords[:,0],y=a_coords[:,1],z=a_coords[:,2],color='grey',
							        i=a_triang[:,0],j=a_triang[:,1],k=a_triang[:,2],
							        opacity=0.1,name="Manual Segmentation")])
	fig.add_trace(go.Scatter3d(x=waikato_path[:,0],y=waikato_path[:,1],z=waikato_path[:,2],
							   mode='markers',marker=dict(size=3,color="red",opacity=0.8),
						       name="Catheter Path"))
	fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=True,
					  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=1),
					  #           xaxis=dict(showbackground=False,tickvals=[],ticktext=[],xaxis_title=''),
					  #           yaxis=dict(showbackground=False,tickvals=[],ticktext=[],yaxis_title=''),
					  #           zaxis=dict(showbackground=False,tickvals=[],ticktext=[],zaxis_title='')
					            )
					  )
	#fig.show()
	plotly.offline.plot(fig,filename=waikato_patients[n]+"/Visualization/LGE-MRI+ProjectedPath.html",auto_open=False)

#plotly.offline.plot(fig,filename=waikato_patients[n]+"/temp.html",auto_open=False)
pd.DataFrame(waikato_path).to_csv(waikato_patients[n]+"/"+waikato_patient_LGEMRI[n]+"_projected_path.csv",header=["x","y","z"],index=False)
