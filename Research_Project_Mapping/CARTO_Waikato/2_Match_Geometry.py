from utils import *

os.chdir("C:/Users/zxio506/Desktop/Atria_Data/CatheterPath/2020_Waikato_PointCloud")

### Waikato Mesh --------------------------------------------------------------------------------------------------
n = 1

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

### Waikato LGE-MRI -----------------------------------------------------------------------------------------------
os.chdir("C:/Users/zxio506/Desktop/Atria_Data/Waikato")

# patient folders
waikato_patient_LGEMRI = ["","00003","00004","00005","","000011"]

# load data
img,lab = np.zeros([640,640,44]),np.zeros([640,640,44])
f1,f2   = os.listdir(waikato_patient_LGEMRI[n]+"/lgemri"),os.listdir(waikato_patient_LGEMRI[n]+"/label")

for i in range(img.shape[2]):
	img[:,:,i] = cv2.imread(waikato_patient_LGEMRI[n]+"/lgemri/"+f1[i],cv2.IMREAD_GRAYSCALE)
	lab[:,:,i] = cv2.imread(waikato_patient_LGEMRI[n]+"/label/"+f2[i],cv2.IMREAD_GRAYSCALE)
	
lab = (lab==4).astype(np.int8)

# sample 3D points from the mesh to produce point cloud
def raster2mesh(mask_temp):

	mask = np.copy(mask_temp)

	# interpolate and create a thin outer shell
	mask_temp = np.zeros_like(mask)
	for i in range(mask.shape[2]):
		mask_temp[:,:,i] = cv2.erode(np.uint8(mask[:,:,i]),np.ones((5,5),np.uint8),iterations=2)

	# compute mesh
	x,y,z = np.where((mask - mask_temp) == 1)
	coords = np.array([x,y,z]).T
	vert,edge,triang = simple_alpha_shape_3D(coords,1)
	
	return(coords,triang)

a_coords,a_triang = raster2mesh(lab)

### Match ----------------------------------------------------------------------------------------------------------

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

# move to same coordinates such that the COMs match
conchull_com    = np.mean(p_conchull,0)
p_conchull[:,0] = p_conchull[:,0] + (com[0] - conchull_com[0])
p_conchull[:,1] = p_conchull[:,1] + (com[1] - conchull_com[1])
p_conchull[:,2] = p_conchull[:,2] + (com[2] - conchull_com[2])






# swap some axis
new_path      = np.copy(path)
new_path[:,0] = path[:,1]
new_path[:,1] = path[:,0]
path          = np.copy(new_path)

# rotate
polar_path          = cartesian2polar(path[:,2],path[:,0],com[2],com[0])
path[:,0],path[:,2] = polar2cartesian(polar_path[0],polar_path[1]+rotate_degree,com[0],com[2])

# rescale to start from (0,0,0)
path[:,0] = path[:,0] - np.min(path[:,0])
path[:,1] = path[:,1] - np.min(path[:,1])
path[:,2] = path[:,2] - np.min(path[:,2])

# resize to same dims as utah, and move to same location
path[:,0]    = path[:,0]/np.max(path[:,0])*(max_x-min_x) + (com[0] - conchull_com[0])
path[:,1]    = path[:,1]/np.max(path[:,1])*(max_y-min_y) + (com[1] - conchull_com[1])
path[:,2]    = path[:,2]/np.max(path[:,2])*(max_z-min_z) + (com[2] - conchull_com[2])

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

path[:,0] -= 10

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

#path[:,1] += 10

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

### Plot -----------------------------------------------------------------------------------------------------------
if True:
	fig = go.Figure(data=[go.Mesh3d(x=a_coords[:,1],y=a_coords[:,0],z=a_coords[:,2],color='grey',
							        i=a_triang[:,1],j=a_triang[:,0],k=a_triang[:,2],
							        opacity=0.1,name="Manual Segmentation")])
	#fig.add_trace(go.Mesh3d(x=p_conchull[:,1],y=p_conchull[:,0],z=p_conchull[:,2],color='pink',
	#						        i=tri_conchull[:,1],j=tri_conchull[:,0],k=tri_conchull[:,2],
	#						        opacity=0.1,name="Mesh"))
	fig.add_trace(go.Scatter3d(x=path[:,1],y=path[:,0],z=path[:,2],
						       #mode='markers',marker=dict(size=3,color=path_dist,colorscale='Jet',opacity=0.8),
                                                       mode='markers',marker=dict(size=3,color='blue',opacity=0.8),
						       name="Catheter Path"))
	fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=True,
					  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=1),
					             xaxis=dict(showbackground=False,tickvals=[]),
					             yaxis=dict(showbackground=False,tickvals=[]),
					             zaxis=dict(showbackground=False,tickvals=[])
					            )
					  )
	#fig.show()
	#plotly.offline.plot(fig,filename="overlay+mesh.html",auto_open=False)
	plotly.offline.plot(fig,filename="C:/Users/zxio506/Desktop/temp.html",auto_open=False)

	
