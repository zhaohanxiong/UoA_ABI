from utils import *

### Set Origin Path
n = 1

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

# patient folders
waikato_patient_LGEMRI = ["","00003","00004","00005","","000011"]

# load data
a_coords = np.array(pd.read_csv(waikato_patients[n]+"/"+waikato_patient_LGEMRI[n]+"_real_path.csv"))

a_vert,_,a_triang = simple_alpha_shape_3D(a_coords,10)

# carto mesh
dat    = scipy.io.loadmat(waikato_patients[n]+"/catheter.mat")
coords = dat['coord']

# only keep points near the outer surface
mesh_coords = a_coords[a_vert,:]
com         = np.mean(a_coords,0)

path_dist = []
for i in range(a_coords.shape[0]):

	temp = ((mesh_coords[:,0]-a_coords[i,0])**2 + (mesh_coords[:,1]-a_coords[i,1])**2 + (mesh_coords[:,2]-a_coords[i,2])**2)**0.5
	
	dist = np.min(temp)
	ind  = np.where(temp==dist)[0][0]
	
	if np.mean(np.abs(a_coords[i] - com)) >= np.mean(np.abs(mesh_coords[ind] - com)):
		path_dist.append(0)
	else:
		path_dist.append(dist)

path_dist         = np.array(path_dist)
a_coords_filtered = a_coords[path_dist < 1,:]

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

# utah center of mass
utah_com = np.array(center_of_mass(lab))

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

# Project To Surface -----------------------------------------------------------------------------------------------
# waikato center of mass
if True:

	path        = np.copy(a_coords_filtered)
	waikato_com = np.array(center_of_mass(lab))

	# project catheter path to waikato data surface
	waikato_path = []
	for i in range(path.shape[0]):
		
		# find closest point in catheter path to carto surface
		path_point  = path[i,:]
		ind         = np.argmin(np.sum((path_point-coords)**2,1)**0.5)
		carto_point = coords[ind,:]

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
os.chdir("C:/Users/zxio506/Desktop")

if True:
	fig = go.Figure(data=[go.Mesh3d(x=a_coords[:,0],y=a_coords[:,1],z=a_coords[:,2],color='grey',
							        i=a_triang[:,0],j=a_triang[:,1],k=a_triang[:,2],
							        opacity=0.1,name="Mesh")])
	fig.add_trace(go.Scatter3d(x=waikato_path[:,0],y=waikato_path[:,1],z=waikato_path[:,2],
							   #mode='markers',marker=dict(size=2,color=path_dist,colorscale='Jet',opacity=0.8),
							   mode='markers',marker=dict(size=3,color="green",opacity=0.5),
						       name="Catheter Path"))
	fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=True,
					  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=1),
					  #           xaxis=dict(showbackground=False,tickvals=[],ticktext=[],xaxis_title=''),
					  #           yaxis=dict(showbackground=False,tickvals=[],ticktext=[],yaxis_title=''),
					  #           zaxis=dict(showbackground=False,tickvals=[],ticktext=[],zaxis_title='')
					            )
					  )
	plotly.offline.plot(fig,filename="LGE-MRI+ProjectedPath.html",auto_open=False)

pd.DataFrame(waikato_path).to_csv(waikato_patient_LGEMRI[n]+"_projected_path.csv",header=["x","y","z"],index=False)
