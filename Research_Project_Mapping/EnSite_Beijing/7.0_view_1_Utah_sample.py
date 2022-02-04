from utils import *
import numpy as np
import cv2
import plotly
import plotly.graph_objects as go
from scipy.ndimage import label
from scipy.ndimage.measurements import center_of_mass

utah_patient = "1421pre" # "1160pre" "1421pre" "0100pre" "02914mo" "10686mo" "10833mo" "13878mo" "08866mo"

# load data
laendo          = np.rollaxis(load_nrrd("Utah_Sample/"+utah_patient+"/laendo.nrrd")//255,0,3)
lawall          = np.rollaxis(load_nrrd("Utah_Sample/"+utah_patient+"/lawall.nrrd")//255,0,3)
laendo_no_veins = np.rollaxis(load_nrrd("Utah_Sample/"+utah_patient+"/laendo_no_veins.nrrd")//255,0,3)

# get veins
utah_PV,utah_PV_names,mask_PV = compute_utah_PV(laendo,lawall,laendo_no_veins)

# sample 3D points from the mesh to produce point cloud
def raster2mesh(mask_temp):

	mask = np.copy(mask_temp)

	# interpolate and create a thin outer shell
	mask_temp = np.zeros_like(mask)
	for i in range(mask.shape[2]):
		mask_temp[:,:,i] = cv2.erode(np.uint8(mask[:,:,i]),np.ones((3,3),np.uint8),iterations=2)

	# compute mesh
	x,y,z = np.where((mask - mask_temp) == 1)
	coords = np.array([x,y,z]).T
	vert,edge,triang = simple_alpha_shape_3D(coords,2)
	
	return(coords,triang)

a_coords,a_triang = raster2mesh(laendo_no_veins)

mask_PV[mask_PV>0] = 1
pv_coords,pv_triang = raster2mesh(mask_PV)

# plot
if True: # True False
	fig = go.Figure(data=[go.Mesh3d(x=a_coords[:,1],y=a_coords[:,0],z=a_coords[:,2],color='grey',
							        i=a_triang[:,1],j=a_triang[:,0],k=a_triang[:,2],
							        opacity=0.1,name="LA Surface")])
	fig.add_trace(go.Mesh3d(x=pv_coords[:,1],y=pv_coords[:,0],z=pv_coords[:,2],color='mediumpurple',
							i=pv_triang[:,1],j=pv_triang[:,0],k=pv_triang[:,2],opacity=0.075,name="Pulmonary Veins"))
	fig.add_trace(go.Scatter3d(x=a_coords[:,1],y=a_coords[:,0],z=a_coords[:,2],
						       mode='markers',marker=dict(size=1,color="black",opacity=0.1)))
	fig.add_trace(go.Scatter3d(x=utah_PV[1,:],y=utah_PV[0,:],z=utah_PV[2,:]*(np.max(a_coords[:,2])//45+1),
						       mode='markers+text',marker=dict(size=5,color="lightpink",symbol='diamond'),
						       text=utah_PV_names,textfont={"size":20,"color":"red"},textposition="top center"))	
	fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
					  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=0.75),
					             xaxis=dict(showbackground=False),xaxis_title='', #,tickvals=[],ticktext=[]
					             yaxis=dict(showbackground=False),yaxis_title='', #,tickvals=[],ticktext=[]
					             zaxis=dict(showbackground=False),zaxis_title='',  #,tickvals=[],ticktext=[]
					            )
					  )
	fig.show()
	
	# save data
	#plotly.offline.plot(fig,filename='Plots/UtahData_'+utah_patient+'.html',auto_open=False)

# manually add PV co-ords
#utah_PV_names = np.append(utah_PV_names,"LSPV")
#utah_PV = np.concatenate((utah_PV.T,np.array([[356,305,32]])), 0).T


if False:
	fig = go.Figure(data=[go.Mesh3d(x=a_coords[:,1],y=a_coords[:,0],z=a_coords[:,2],color='grey',
							        i=a_triang[:,1],j=a_triang[:,0],k=a_triang[:,2],
							        opacity=0.1,name="LA Surface")])
	fig.add_trace(go.Mesh3d(x=pv_coords[:,1],y=pv_coords[:,0],z=pv_coords[:,2],color='mediumpurple',
							i=pv_triang[:,1],j=pv_triang[:,0],k=pv_triang[:,2],opacity=0.075,name="Pulmonary Veins"))
	fig.add_trace(go.Scatter3d(x=a_coords[:,1],y=a_coords[:,0],z=a_coords[:,2],
						       mode='markers',marker=dict(size=1,color="black",opacity=0.1)))
	fig.add_trace(go.Scatter3d(x=utah_PV[1,:],y=utah_PV[0,:],z=utah_PV[2,:]*(np.max(a_coords[:,2])//45+1),
						       mode='markers+text',marker=dict(size=5,color="lightpink",symbol='diamond'),
						       text=utah_PV_names,textfont={"size":20,"color":"red"},textposition="top center"))

	#PV_com = np.mean(utah_PV,1)						   					
	#fig.add_trace(go.Scatter3d(x=[PV_com[1],utah_PV[1,0]],y=[PV_com[0],utah_PV[0,0]],z=[PV_com[2],utah_PV[2,0]],line=dict(color="red",width=5)))
	#fig.add_trace(go.Scatter3d(x=[PV_com[1],utah_PV[1,1]],y=[PV_com[0],utah_PV[0,1]],z=[PV_com[2],utah_PV[2,1]],line=dict(color="red",width=5)))
	#fig.add_trace(go.Scatter3d(x=[PV_com[1],utah_PV[1,2]],y=[PV_com[0],utah_PV[0,2]],z=[PV_com[2],utah_PV[2,2]],line=dict(color="red",width=5)))
	#fig.add_trace(go.Scatter3d(x=[PV_com[1],utah_PV[1,3]],y=[PV_com[0],utah_PV[0,3]],z=[PV_com[2],utah_PV[2,3]],line=dict(color="red",width=5)))
	
	fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",showlegend=False,
					  scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=0.75),
					             xaxis=dict(showbackground=False),xaxis_title='', #,tickvals=[],ticktext=[]
					             yaxis=dict(showbackground=False),yaxis_title='', #,tickvals=[],ticktext=[]
					             zaxis=dict(showbackground=False),zaxis_title='',  #,tickvals=[],ticktext=[]
					            )
					  )
	fig.show()
